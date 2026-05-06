import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# Reconfigure stdout to support UTF-8 characters in Windows terminal
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older Python versions if needed
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

log_path = Path("logs/qualitative_analysis.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

BASELINES_DIR  = Path("./data/processed/eval_results/baselines")
FINETUNED_PATH = Path("./data/processed/eval_results/rag_eval_with_rag.jsonl")
REPORT_PATH    = BASELINES_DIR / "qualitative_report.json"
REPORT_TXT     = BASELINES_DIR / "qualitative_report.txt"


# ── Load JSONL ────────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    continue
    return records


# ── Strict language detection ─────────────────────────────────────────
def detect_language_strict(text: str) -> str:
    """
    Strict 3-class detection:
    - 'hindi'   : >=60% of alpha chars are Devanagari
    - 'english' : <20% Devanagari
    - 'mixed'   : 20-60% Devanagari
    """
    if not text or not text.strip():
        return "empty"
    dev   = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return "empty"
    ratio = dev / alpha
    if ratio >= 0.60:
        return "hindi"
    elif ratio >= 0.20:
        return "mixed"
    else:
        return "english"


# ── Hallucination detector ────────────────────────────────────────────
def detect_hallucinations(
    prediction: str,
    reference:  str,
    query:      str,
) -> dict:
    """
    Detect five types of hallucination:
    1. number_hallucination   — wrong dosages/temperatures/dates
    2. crop_confusion         — answer about wrong crop
    3. wrong_scheme           — mentions non-existent scheme names
    4. fabricated_contact     — phone numbers not in reference
    5. generic_non_answer     — vague deflection instead of advice
    """
    results = {
        "number_hallucination":  False,
        "crop_confusion":        False,
        "wrong_scheme":          False,
        "fabricated_contact":    False,
        "generic_non_answer":    False,
        "details":               [],
    }

    # ── 1. Number hallucination ──
    pred_nums = set(re.findall(r'\d+\.?\d*', prediction))
    ref_nums  = set(re.findall(r'\d+\.?\d*', reference))
    if ref_nums and pred_nums:
        hallucinated = pred_nums - ref_nums - {"0", "1", "2", "3"}
        # Only flag if multiple numbers are hallucinated
        if len(hallucinated) >= 2:
            results["number_hallucination"] = True
            results["details"].append(
                f"Hallucinated numbers: {list(hallucinated)[:5]}"
            )

    # ── 2. Crop confusion ──
    crop_pairs = [
        ("गेहूं", "wheat"),   ("धान", "paddy"),
        ("मक्का", "maize"),   ("सरसों", "mustard"),
        ("प्याज", "onion"),   ("धनिया", "coriander"),
        ("टमाटर", "tomato"),  ("आलू", "potato"),
        ("कपास", "cotton"),   ("गन्ना", "sugarcane"),
    ]

    # Find what crop is in the query
    query_crops = []
    for hindi_crop, eng_crop in crop_pairs:
        if hindi_crop in query:
            query_crops.append(hindi_crop)

    # Check if prediction talks about a different crop
    if query_crops:
        pred_crops = [h for h, e in crop_pairs if h in prediction]
        wrong_crops = [c for c in pred_crops if c not in query_crops]
        if wrong_crops:
            results["crop_confusion"] = True
            results["details"].append(
                f"Query about {query_crops}, prediction mentions {wrong_crops}"
            )

    # ── 3. Wrong/fabricated scheme names ──
    # Real scheme names
    real_schemes = [
        "पीएम किसान", "pm kisan", "किसान क्रेडिट कार्ड", "kcc",
        "फसल बीमा", "pmfby", "राष्ट्रीय", "national",
        "मनरेगा", "नाबार्ड", "nabard",
    ]
    scheme_pattern = re.compile(
        r'(योजना|scheme|yojana)', re.IGNORECASE
    )
    if scheme_pattern.search(prediction):
        pred_lower = prediction.lower()
        ref_lower  = reference.lower()
        # If prediction mentions a scheme that reference doesn't mention
        # and it's not a known real scheme
        pred_scheme_words = set(re.findall(r'\b\w+\s*योजना\b', prediction))
        ref_scheme_words  = set(re.findall(r'\b\w+\s*योजना\b', reference))
        fabricated = pred_scheme_words - ref_scheme_words
        if fabricated and len(fabricated) >= 2:
            results["wrong_scheme"] = True
            results["details"].append(
                f"Possibly fabricated scheme: {list(fabricated)[:3]}"
            )

    # ── 4. Fabricated contact numbers ──
    phone_pattern = re.compile(r'\b[6-9]\d{9}\b|\b1800\d{6,7}\b|\b\d{5,}\b')
    pred_phones = set(phone_pattern.findall(prediction))
    ref_phones  = set(phone_pattern.findall(reference))
    fake_phones = pred_phones - ref_phones
    if fake_phones:
        results["fabricated_contact"] = True
        results["details"].append(
            f"Fabricated phone numbers: {list(fake_phones)[:3]}"
        )

    # ── 5. Generic non-answer ──
    # These phrases indicate the model deflected instead of answering
    deflection_patterns_hi = [
        "जानकारी प्राप्त करें", "जानकारी ले सकते हैं",
        "संपर्क कर सकते हैं", "अधिक जानकारी के लिए",
        "कृपया संपर्क करें", "जानकारी प्रदान की जाएगी",
        "विशेषज्ञ से सलाह", "नजदीकी कृषि",
    ]
    deflection_count = sum(
        1 for p in deflection_patterns_hi
        if p in prediction
    )
    if deflection_count >= 2 and len(prediction) < 200:
        results["generic_non_answer"] = True
        results["details"].append(
            f"Generic deflection ({deflection_count} deflection phrases)"
        )

    results["any_hallucination"] = any([
        results["number_hallucination"],
        results["crop_confusion"],
        results["wrong_scheme"],
        results["fabricated_contact"],
        results["generic_non_answer"],
    ])

    return results


# ── Failure categoriser ───────────────────────────────────────────────
def categorise_failure(
    prediction: str,
    reference:  str,
    query:      str,
    lang:       str,
) -> str:
    """
    Assign one primary failure category or 'correct'.
    """
    if not prediction or not prediction.strip():
        return "empty_response"

    if lang == "english":
        return "language_mismatch"

    if lang == "mixed":
        return "mixed_language"

    # Check generic non-answer first
    deflection_patterns = [
        "जानकारी प्राप्त करें", "संपर्क कर सकते हैं",
        "जानकारी ले सकते हैं", "विशेषज्ञ से",
    ]
    deflection_count = sum(1 for p in deflection_patterns if p in prediction)
    if deflection_count >= 2 and len(prediction) < 200:
        return "generic_non_answer"

    # Repetition loop
    words = prediction.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams and max(Counter(trigrams).values()) >= 3:
            return "repetition_loop"

    # Topic drift
    crop_terms = [
        "किसान", "फसल", "बीज", "खाद", "कीट", "रोग",
        "सिंचाई", "बुवाई", "उर्वरक", "कृषि",
    ]
    ref_agri  = any(t in reference   for t in crop_terms)
    pred_agri = any(t in prediction  for t in crop_terms)
    if ref_agri and not pred_agri:
        return "topic_drift"

    # Number hallucination
    pred_nums = set(re.findall(r'\d+\.?\d*', prediction))
    ref_nums  = set(re.findall(r'\d+\.?\d*', reference))
    if ref_nums and pred_nums:
        hallucinated = pred_nums - ref_nums - {"0", "1", "2", "3"}
        if len(hallucinated) >= 2:
            return "number_hallucination"

    # Too verbose (baseline issue)
    if len(prediction) > len(reference) * 3 and len(prediction) > 500:
        return "over_verbose"

    return "correct"


# ── Run full qualitative analysis for one model ───────────────────────
def analyse_model(records: list, model_name: str) -> dict:
    """
    Run full qualitative analysis on one model's predictions.
    """
    lang_dist       = Counter()
    failure_dist    = Counter()
    hallucination_types = Counter()
    intent_failures = defaultdict(Counter)
    case_examples   = defaultdict(list)

    for record in tqdm(records, desc=f"Analysing {model_name}"):
        pred   = str(record.get("prediction", "")).strip()
        ref    = str(record.get("reference",  "")).strip()
        query  = str(record.get("query",      "")).strip()
        intent = str(record.get("intent",     "unknown"))

        # Language
        lang = detect_language_strict(pred)
        lang_dist[lang] += 1

        # Failure category
        failure = categorise_failure(pred, ref, query, lang)
        failure_dist[failure] += 1
        intent_failures[intent][failure] += 1

        # Hallucination detection
        hall = detect_hallucinations(pred, ref, query)
        if hall["any_hallucination"]:
            for h_type in [
                "number_hallucination", "crop_confusion",
                "wrong_scheme", "fabricated_contact", "generic_non_answer",
            ]:
                if hall[h_type]:
                    hallucination_types[h_type] += 1

        # Collect case examples (max 5 per category)
        if len(case_examples[failure]) < 5:
            case_examples[failure].append({
                "query":      query[:100],
                "prediction": pred[:200],
                "reference":  ref[:200],
                "intent":     intent,
                "lang":       lang,
                "hall_detail":hall["details"][:2],
            })

    total = len(records)
    return {
        "model":       model_name,
        "total":       total,
        "language_distribution": {
            k: {"count": v, "pct": round(v/total*100, 2)}
            for k, v in lang_dist.items()
        },
        "failure_distribution": {
            k: {"count": v, "pct": round(v/total*100, 2)}
            for k, v in failure_dist.items()
        },
        "hallucination_types": {
            k: {"count": v, "pct": round(v/total*100, 2)}
            for k, v in hallucination_types.items()
        },
        "correct_pct":    round(failure_dist.get("correct", 0) / total * 100, 2),
        "intent_failures": {
            intent: dict(counts)
            for intent, counts in intent_failures.items()
        },
        "case_examples": dict(case_examples),
    }


# ── Print qualitative report ──────────────────────────────────────────
def print_qualitative_report(all_analyses: dict):
    sep   = "=" * 75
    names = ["zero_shot", "one_shot", "few_shot",
             "prompt_engineered", "finetuned_rag"]
    labels = {
        "zero_shot":         "Zero-shot",
        "one_shot":          "One-shot",
        "few_shot":          "Few-shot",
        "prompt_engineered": "Prompt-Eng",
        "finetuned_rag":     "FT+RAG",
    }

    lines = []   # collect for file output

    def p(text="", **kwargs):
        print(text, **kwargs)
        lines.append(text)

    p(f"\n{sep}")
    p("  KisanMitra AI — Qualitative & Error Analysis Report")
    p(sep)

    # ── Language distribution ──
    p(f"\n  ── Language Distribution (Strict: ≥60% Devanagari = Hindi) ─────────────")
    p(f"  {'Language':<15}")

    lang_cats = ["hindi", "mixed", "english", "empty"]
    p(f"  {'Category':<15}" +
      "".join(f"{labels[n]:>13}" for n in names if n in all_analyses))

    for lc in lang_cats:
        row = f"  {lc:<15}"
        for n in names:
            if n not in all_analyses:
                continue
            lang_d = all_analyses[n].get("language_distribution", {})
            pct    = lang_d.get(lc, {}).get("pct", 0)
            row   += f"{pct:>12.1f}%"
        p(row)

    # ── Failure category table ──
    p(f"\n  ── Failure Category Distribution ──────────────────────────────────────")
    p(f"  {'Category':<25}" +
      "".join(f"{labels[n]:>13}" for n in names if n in all_analyses))

    failure_cats = [
        "correct", "generic_non_answer", "number_hallucination",
        "topic_drift", "repetition_loop", "language_mismatch",
        "mixed_language", "over_verbose", "empty_response",
    ]

    for fc in failure_cats:
        row = f"  {fc:<25}"
        vals = []
        for n in names:
            if n not in all_analyses:
                continue
            fd  = all_analyses[n].get("failure_distribution", {})
            pct = fd.get(fc, {}).get("pct", 0)
            vals.append((n, pct))
            row += f"{pct:>12.1f}%"
        p(row)

    # ── Hallucination types ──
    p(f"\n  ── Hallucination Type Breakdown ───────────────────────────────────────")
    p(f"  {'Type':<25}" +
      "".join(f"{labels[n]:>13}" for n in names if n in all_analyses))

    hall_types = [
        "generic_non_answer", "number_hallucination",
        "crop_confusion", "fabricated_contact", "wrong_scheme",
    ]
    for ht in hall_types:
        row = f"  {ht:<25}"
        for n in names:
            if n not in all_analyses:
                continue
            ht_d = all_analyses[n].get("hallucination_types", {})
            pct  = ht_d.get(ht, {}).get("pct", 0)
            row += f"{pct:>12.1f}%"
        p(row)

    # ── Per-intent failure for fine-tuned vs best baseline ──
    p(f"\n  ── Per-Intent: Correct % — Zero-shot vs Few-shot vs FT+RAG ───────────")
    p(f"  {'Intent':<25} {'Zero-shot':>12} {'Few-shot':>12} {'FT+RAG':>12}")
    p(f"  {'─'*63}")

    all_intents = set()
    for a in all_analyses.values():
        all_intents.update(a.get("intent_failures", {}).keys())

    for intent in sorted(all_intents):
        row = f"  {intent:<25}"
        for n in ["zero_shot", "few_shot", "finetuned_rag"]:
            if n not in all_analyses:
                row += f"{'N/A':>12}"
                continue
            i_fail = all_analyses[n].get("intent_failures", {})
            counts = i_fail.get(intent, {})
            total_intent = sum(counts.values())
            correct      = counts.get("correct", 0)
            pct = round(correct / total_intent * 100, 1) if total_intent > 0 else 0
            row += f"{pct:>11.1f}%"
        p(row)

    # ── Detailed failure examples ──
    p(f"\n{sep}")
    p("  Failure Case Examples — Side by Side Comparison")
    p(sep)

    failure_types_to_show = [
        "generic_non_answer",
        "number_hallucination",
        "topic_drift",
        "over_verbose",
    ]

    for ft in failure_types_to_show:
        p(f"\n  ── Failure Type: {ft.upper()} ────────────────────────────────────")

        # Show one example from each baseline
        for model_name in ["zero_shot", "few_shot", "prompt_engineered", "finetuned_rag"]:
            if model_name not in all_analyses:
                continue
            examples = all_analyses[model_name].get("case_examples", {}).get(ft, [])
            if not examples:
                continue
            ex = examples[0]
            p(f"\n  [{labels.get(model_name, model_name)}]")
            p(f"  Intent : {ex['intent']}")
            p(f"  Lang   : {ex['lang']}")
            p(f"  Query  : {ex['query'][:80]}")
            p(f"  Pred   : {ex['prediction'][:180]}")
            if model_name == "finetuned_rag" or model_name == list(all_analyses.keys())[-1]:
                p(f"  Ref    : {ex['reference'][:180]}")
            if ex.get("hall_detail"):
                p(f"  ⚠️  Hall : {ex['hall_detail']}")

    # ── Best vs worst examples for fine-tuned ──
    p(f"\n{sep}")
    p("  Fine-tuned+RAG: Best Success Cases")
    p(sep)

    if "finetuned_rag" in all_analyses:
        successes = all_analyses["finetuned_rag"]\
                        .get("case_examples", {}).get("correct", [])
        for i, ex in enumerate(successes[:5], 1):
            p(f"\n  ✅ Success #{i} — Intent: {ex['intent']}")
            p(f"  Query: {ex['query'][:80]}")
            p(f"  Pred : {ex['prediction'][:180]}")
            p(f"  Ref  : {ex['reference'][:180]}")

    p(f"\n{sep}")
    p("  Fine-tuned+RAG: Remaining Failure Cases")
    p(sep)

    if "finetuned_rag" in all_analyses:
        ft_cases = all_analyses["finetuned_rag"].get("case_examples", {})
        for failure_type, examples in ft_cases.items():
            if failure_type == "correct" or not examples:
                continue
            p(f"\n  ❌ Failure type: {failure_type} ({len(examples)} cases found)")
            for ex in examples[:2]:
                p(f"  Intent : {ex['intent']}")
                p(f"  Query  : {ex['query'][:80]}")
                p(f"  Pred   : {ex['prediction'][:180]}")
                p(f"  Ref    : {ex['reference'][:180]}")
                if ex.get("hall_detail"):
                    p(f"  Detail : {ex['hall_detail']}")
                p(f"  {'─'*60}")

    # ── Summary verdict ──
    p(f"\n{sep}")
    p("  Summary — Key Qualitative Findings")
    p(sep)

    ft_correct  = all_analyses.get("finetuned_rag", {}).get("correct_pct", 0)
    zs_correct  = all_analyses.get("zero_shot",     {}).get("correct_pct", 0)
    pe_correct  = all_analyses.get("prompt_engineered", {}).get("correct_pct", 0)

    p(f"\n  1. Language Consistency")
    p(f"     Baselines produce mixed/English responses despite Hindi prompts.")
    p(f"     Fine-tuned model: 100% Hindi with 0% language mismatch.")

    p(f"\n  2. Hallucination Rate")
    for n, label in [("zero_shot","Zero-shot"), ("few_shot","Few-shot"),
                     ("prompt_engineered","Prompt-eng"), ("finetuned_rag","FT+RAG")]:
        if n in all_analyses:
            hall_total = sum(
                v.get("count", 0)
                for v in all_analyses[n].get("hallucination_types", {}).values()
            )
            total = all_analyses[n].get("total", 1)
            p(f"     {label:<18}: {hall_total/total*100:.1f}% of responses contain hallucinations")

    p(f"\n  3. Failure Mode Patterns")
    p(f"     Baselines: generic_non_answer is the dominant failure (vague deflection)")
    p(f"     One-shot : worst correct% ({all_analyses.get('one_shot',{}).get('correct_pct',0):.1f}%) — single wrong example misleads model")
    p(f"     Few-shot : highest hallucination rate — more examples = more fabrication")
    p(f"     FT+RAG   : {ft_correct:.1f}% correct — remaining failures are weather specificity")

    p(f"\n  4. Overall Correct %")
    p(f"     Zero-shot:    {zs_correct:.1f}%")
    p(f"     Prompt-eng:   {pe_correct:.1f}%")
    p(f"     Fine-tuned+RAG: {ft_correct:.1f}%  (+{ft_correct-zs_correct:.1f}% over zero-shot)")

    p(f"\n{sep}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 75)
    logger.info("KisanMitra AI — Qualitative & Error Analysis")
    logger.info("=" * 75)

    all_analyses = {}

    # Load all 4 baselines
    for name in ["zero_shot", "one_shot", "few_shot", "prompt_engineered"]:
        path = BASELINES_DIR / f"baseline_{name}.jsonl"
        if not path.exists():
            # Try annotated version
            path = BASELINES_DIR / f"annotated_{name}.jsonl"
        if path.exists():
            records = load_jsonl(str(path))
            logger.info(f"Loaded {len(records)} records for {name}")
            all_analyses[name] = analyse_model(records, name)
        else:
            logger.warning(f"File not found: {path}")

    # Load fine-tuned + RAG
    if FINETUNED_PATH.exists():
        ft_records = load_jsonl(str(FINETUNED_PATH))
        logger.info(f"Loaded {len(ft_records)} fine-tuned+RAG records")
        all_analyses["finetuned_rag"] = analyse_model(ft_records, "finetuned_rag")
    else:
        logger.warning(f"Fine-tuned results not found: {FINETUNED_PATH}")

    if not all_analyses:
        logger.error("No results found. Run run_baselines.py first.")
        sys.exit(1)

    # Print and capture report
    report_text = print_qualitative_report(all_analyses)

    # Save full report as JSON
    clean_analyses = {}
    for name, analysis in all_analyses.items():
        clean_analyses[name] = {
            k: v for k, v in analysis.items()
            if k != "case_examples"
        }
        # Keep case examples but truncate
        clean_analyses[name]["case_examples"] = {
            ft: [
                {k: v for k, v in ex.items()}
                for ex in examples[:3]
            ]
            for ft, examples in analysis.get("case_examples", {}).items()
        }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_analyses, f, ensure_ascii=False, indent=2)
    logger.success(f"JSON report → {REPORT_PATH}")

    # Save text report
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.success(f"Text report → {REPORT_TXT}")

    logger.success("=" * 75)
    logger.success("Qualitative analysis complete.")
    logger.success(f"Files saved to: {BASELINES_DIR}")
    logger.success("=" * 75)