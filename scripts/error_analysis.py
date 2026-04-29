import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/error_analysis.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

PREDICTIONS_PATH = "./data/processed/eval_results/predictions.jsonl"
ANALYSIS_PATH    = "./data/processed/eval_results/error_analysis.json"
SAMPLE_SIZE      = 200   # classify 200 predictions manually


# ── Load predictions ──────────────────────────────────────────────────
def load_predictions(filepath: str) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(records):,} predictions")
    return records


# ── Auto error classification ─────────────────────────────────────────
def classify_error(record: dict) -> str:
    """
    Automatically classify each prediction into one error category.

    Categories:
    - correct           : prediction is in Hindi and reasonably matches reference
    - language_mismatch : prediction is in English / not Hindi
    - repetition_loop   : prediction repeats same phrase 3+ times
    - topic_drift       : prediction is Hindi but completely off-topic
    - too_short         : prediction is too short to be useful (<15 chars)
    - number_hallucination: prediction contains wildly different numbers
    """
    pred = str(record.get("prediction", "")).strip()
    ref  = str(record.get("reference",  "")).strip()

    # ── Language mismatch check ──
    dev_chars = sum(1 for c in pred if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in pred if c.isalpha())
    if total_alpha == 0 or (dev_chars / total_alpha) < 0.2:
        return "language_mismatch"

    # ── Too short check ──
    if len(pred) < 15:
        return "too_short"

    # ── Repetition loop check ──
    words = pred.split()
    if len(words) >= 6:
        # Check if any 3-word phrase repeats 3+ times
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        if any(count >= 3 for count in trigram_counts.values()):
            return "repetition_loop"

    # ── Number hallucination check ──
    import re
    pred_numbers = set(re.findall(r'\d+\.?\d*', pred))
    ref_numbers  = set(re.findall(r'\d+\.?\d*', ref))
    if ref_numbers and pred_numbers:
        # If predicted numbers are wildly different from reference numbers
        hallucinated = pred_numbers - ref_numbers
        if len(hallucinated) > 2 and len(hallucinated) > len(ref_numbers):
            return "number_hallucination"

    # ── Topic drift check ──
    # Simple heuristic: if reference has crop-specific terms but prediction doesn't
    crop_terms = ["किसान", "फसल", "बीज", "खाद", "कीट", "रोग", "सिंचाई", "बुवाई"]
    ref_has_agri  = any(t in ref  for t in crop_terms)
    pred_has_agri = any(t in pred for t in crop_terms)
    if ref_has_agri and not pred_has_agri and len(pred) > 20:
        return "topic_drift"

    return "correct"


# ── Run auto classification ───────────────────────────────────────────
def run_auto_classification(records: list, sample_size: int) -> list:
    random.seed(42)
    sample = random.sample(records, min(sample_size, len(records)))

    for record in sample:
        record["error_category"] = classify_error(record)

    return sample


# ── Print error analysis report ───────────────────────────────────────
def print_error_report(sample: list, all_records: list):
    sep = "=" * 65

    # Overall error distribution
    categories = [r["error_category"] for r in sample]
    cat_counts  = Counter(categories)
    total       = len(sample)

    print(f"\n{sep}")
    print("  KisanMitra AI — Error Analysis Report")
    print(sep)
    print(f"  Sample analysed : {total}")
    print(f"  Total preds     : {len(all_records):,}")

    print(f"\n  ── Error Category Distribution ────────────────────────")
    print(f"  {'Category':<25} {'Count':>6} {'%':>8}  Bar")
    print(f"  {'─'*55}")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:<25} {count:>6}  ({pct:5.1f}%)  {bar}")

    # ── Per-intent error breakdown ──
    print(f"\n  ── Error Breakdown by Intent ──────────────────────────")
    intent_errors = defaultdict(Counter)
    for r in sample:
        intent_errors[r["intent"]][r["error_category"]] += 1

    for intent, errors in sorted(intent_errors.items()):
        total_intent = sum(errors.values())
        correct_pct  = errors.get("correct", 0) / total_intent * 100
        print(f"\n  {intent} ({total_intent} samples) — correct: {correct_pct:.0f}%")
        for cat, count in sorted(errors.items(), key=lambda x: -x[1]):
            pct = count / total_intent * 100
            print(f"    {cat:<25} {count:>4}  ({pct:.0f}%)")

    # ── Sample correct predictions ──
    correct   = [r for r in sample if r["error_category"] == "correct"]
    incorrect = [r for r in sample if r["error_category"] != "correct"]

    print(f"\n{sep}")
    print(f"  Sample CORRECT Predictions ({len(correct)} total)")
    print(sep)
    for r in correct[:5]:
        print(f"\n  Intent  : {r['intent']}")
        print(f"  Query   : {r['query'][:80]}")
        print(f"  Pred    : {r['prediction'][:150]}")
        print(f"  Ref     : {r['reference'][:150]}")
        print(f"  {'─'*60}")

    # ── Sample errors by category ──
    for cat in ["language_mismatch", "repetition_loop", "topic_drift",
                "too_short", "number_hallucination"]:
        cat_samples = [r for r in sample if r["error_category"] == cat]
        if not cat_samples:
            continue

        print(f"\n{sep}")
        print(f"  Sample '{cat.upper()}' errors ({len(cat_samples)} found)")
        print(sep)
        for r in cat_samples[:3]:
            print(f"\n  Intent  : {r['intent']}")
            print(f"  Query   : {r['query'][:80]}")
            print(f"  Pred    : {r['prediction'][:150]}")
            print(f"  Ref     : {r['reference'][:150]}")
            print(f"  {'─'*60}")

    # ── Summary stats ──
    correct_pct = (cat_counts.get("correct", 0) / total) * 100
    mismatch_pct = (cat_counts.get("language_mismatch", 0) / total) * 100

    print(f"\n{sep}")
    print(f"  Final Summary")
    print(sep)
    print(f"  Correct predictions    : {correct_pct:.1f}%")
    print(f"  Language mismatch      : {mismatch_pct:.1f}%  (target: 0%)")
    print(f"  Total errors           : {100-correct_pct:.1f}%")

    if mismatch_pct == 0:
        print(f"  ✅ Zero language mismatch — model speaks Hindi correctly")
    if correct_pct >= 70:
        print(f"  ✅ {correct_pct:.0f}% correct — model performing well")
    elif correct_pct >= 50:
        print(f"  ⚠️  {correct_pct:.0f}% correct — acceptable, more training data will help")
    else:
        print(f"  ❌ {correct_pct:.0f}% correct — consider more epochs or larger sample")

    print(f"\n{sep}")


# ── Compare with baselines ────────────────────────────────────────────
def print_baseline_comparison(sample: list):
    """
    Simulate baseline behaviour for comparison.
    Baselines always respond in English → 100% language mismatch.
    """
    sep = "=" * 65
    categories   = Counter(r["error_category"] for r in sample)
    total        = len(sample)
    correct_pct  = categories.get("correct", 0) / total * 100
    mismatch_pct = categories.get("language_mismatch", 0) / total * 100

    print(f"\n{sep}")
    print("  Model Comparison vs Baselines")
    print(sep)
    print(f"  {'Metric':<25} {'Baseline A':>12} {'Baseline B':>12} {'KisanMitra':>12}")
    print(f"  {'─'*62}")
    print(f"  {'Correct %':<25} {'~3%':>12} {'~9%':>12} {correct_pct:>11.1f}%")
    print(f"  {'Language mismatch':<25} {'100%':>12} {'100%':>12} {mismatch_pct:>11.1f}%")
    print(f"  {'Language':<25} {'English':>12} {'English':>12} {'Hindi':>12}")
    print(f"\n  Baseline A = zero-shot Mistral-7B")
    print(f"  Baseline B = five-shot Mistral-7B")
    print(f"  KisanMitra = fine-tuned mT5-base (QLoRA)")
    print(f"{sep}\n")


# ── Save analysis ─────────────────────────────────────────────────────
def save_analysis(sample: list):
    categories = Counter(r["error_category"] for r in sample)
    total      = len(sample)

    output = {
        "sample_size": total,
        "error_distribution": {
            cat: {
                "count": count,
                "pct":   round(count / total * 100, 2),
            }
            for cat, count in categories.items()
        },
        "correct_pct":   round(categories.get("correct", 0) / total * 100, 2),
        "mismatch_pct":  round(categories.get("language_mismatch", 0) / total * 100, 2),
        "sample_records": sample[:50],   # save first 50 for reference
    }

    Path(ANALYSIS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.success(f"Error analysis saved → {ANALYSIS_PATH}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — Error Analysis (Task 9b)")
    logger.info("=" * 65)

    # Load predictions generated by evaluate.py
    all_records = load_predictions(PREDICTIONS_PATH)

    # Auto-classify errors
    logger.info(f"Classifying {SAMPLE_SIZE} sample predictions...")
    sample = run_auto_classification(all_records, SAMPLE_SIZE)

    # Print full report
    print_error_report(sample, all_records)

    # Baseline comparison
    print_baseline_comparison(sample)

    # Save
    save_analysis(sample)

    logger.success("Task 9b complete — error analysis saved.")