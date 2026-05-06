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

log_path = Path("logs/baseline_metrics.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

BASELINES_DIR  = Path("./data/processed/eval_results/baselines")
FINETUNED_PATH = Path("./data/processed/eval_results/rag_eval_with_rag.jsonl")
RESULTS_DIR    = Path("./data/processed/eval_results/baselines")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Load JSONL ─────────────────────────────────────────────────────────
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


# ── Align records by query ────────────────────────────────────────────
def align_records(baseline_records: list, finetuned_records: list) -> tuple:
    """
    Find common queries between baseline and fine-tuned results
    so all models are evaluated on the exact same samples.
    """
    # Build lookup by query text for fine-tuned
    ft_lookup = {}
    for r in finetuned_records:
        q = str(r.get("query", "")).strip()[:100]
        ft_lookup[q] = r

    aligned_baseline  = []
    aligned_finetuned = []

    for r in baseline_records:
        q = str(r.get("query", "")).strip()[:100]
        if q in ft_lookup:
            aligned_baseline.append(r)
            aligned_finetuned.append(ft_lookup[q])

    logger.info(
        f"Aligned {len(aligned_baseline)} common records "
        f"out of {len(baseline_records)} baseline records"
    )
    return aligned_baseline, aligned_finetuned


# ── BLEU ──────────────────────────────────────────────────────────────
def compute_bleu(predictions: list, references: list) -> dict:
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize="char")
    score = bleu.corpus_score(predictions, [references])
    return {
        "bleu4": round(score.score,           4),
        "bleu1": round(score.precisions[0],   4),
        "bleu2": round(score.precisions[1],   4),
        "bleu3": round(score.precisions[2],   4),
        "bp":    round(score.bp,              4),
    }


# ── chrF ──────────────────────────────────────────────────────────────
def compute_chrf(predictions: list, references: list) -> dict:
    from sacrebleu.metrics import CHRF
    chrf  = CHRF()
    score = chrf.corpus_score(predictions, [references])
    return {"chrf": round(score.score, 4)}


# ── ROUGE ─────────────────────────────────────────────────────────────
def compute_rouge(predictions: list, references: list) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(np.mean(r1) * 100, 4),
        "rouge2": round(np.mean(r2) * 100, 4),
        "rougeL": round(np.mean(rL) * 100, 4),
    }


# ── Language metrics ──────────────────────────────────────────────────
def compute_language_stats(predictions: list) -> dict:
    hindi   = 0
    english = 0
    mixed   = 0
    empty   = 0

    for pred in predictions:
        if not pred or not pred.strip():
            empty += 1
            continue
        has_dev = any('\u0900' <= c <= '\u097F' for c in pred)
        has_lat = any(c.isascii() and c.isalpha() for c in pred)
        if has_dev and has_lat:
            mixed += 1
        elif has_dev:
            hindi += 1
        else:
            english += 1

    total = len(predictions)
    return {
        "hindi_pct":   round(hindi   / total * 100, 2),
        "english_pct": round(english / total * 100, 2),
        "mixed_pct":   round(mixed   / total * 100, 2),
        "empty_pct":   round(empty   / total * 100, 2),
        "mismatch_pct":round((english + empty) / total * 100, 2),
    }


# ── Error classification ──────────────────────────────────────────────
def classify_error(prediction: str, reference: str) -> str:
    if not prediction or not prediction.strip():
        return "empty_response"

    dev   = sum(1 for c in prediction if '\u0900' <= c <= '\u097F')
    total = sum(1 for c in prediction if c.isalpha())
    if total > 0 and (dev / total) < 0.2:
        return "language_mismatch"

    if len(prediction.strip()) < 15:
        return "too_short"

    words = prediction.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams and max(Counter(trigrams).values()) >= 3:
            return "repetition_loop"

    pred_nums = set(re.findall(r'\d+\.?\d*', prediction))
    ref_nums  = set(re.findall(r'\d+\.?\d*', reference))
    if ref_nums and pred_nums:
        hallucinated = pred_nums - ref_nums
        if len(hallucinated) > 2 and len(hallucinated) > len(ref_nums):
            return "number_hallucination"

    crop_terms = [
        "किसान", "फसल", "बीज", "खाद", "कीट", "रोग",
        "सिंचाई", "बुवाई", "उर्वरक", "कृषि", "विभाग",
        "संपर्क", "पानी", "छिड़काव", "ग्राम", "लीटर",
    ]
    ref_agri  = any(t in reference   for t in crop_terms)
    pred_agri = any(t in prediction  for t in crop_terms)
    if ref_agri and not pred_agri and len(prediction) > 20:
        return "topic_drift"

    return "correct"


# ── Compute full metrics for one model ───────────────────────────────
def compute_all_metrics(records: list, name: str) -> dict:
    predictions = [str(r.get("prediction", "")).strip() for r in records]
    references  = [str(r.get("reference",  "")).strip() for r in records]
    intents     = [str(r.get("intent",     "unknown"))  for r in records]
    latencies   = [int(r.get("latency_ms", 0))          for r in records]

    # Corpus-level metrics
    bleu  = compute_bleu(predictions,  references)
    chrf  = compute_chrf(predictions,  references)
    rouge = compute_rouge(predictions, references)
    lang  = compute_language_stats(predictions)

    # Per-record error classification
    errors = [classify_error(p, r) for p, r in zip(predictions, references)]
    error_counts = Counter(errors)
    total        = len(records)

    # Per-intent breakdown
    intent_data = defaultdict(lambda: {"correct": 0, "total": 0})
    for r, e in zip(records, errors):
        intent = r.get("intent", "unknown")
        intent_data[intent]["total"]   += 1
        if e == "correct":
            intent_data[intent]["correct"] += 1

    per_intent = {
        intent: {
            "correct_pct": round(v["correct"] / v["total"] * 100, 1),
            "count":       v["total"],
        }
        for intent, v in intent_data.items()
    }

    # Length stats
    pred_lens = [len(p) for p in predictions]
    ref_lens  = [len(r) for r in references]

    return {
        "model":       name,
        "n_samples":   total,
        # Core metrics
        "bleu4":       bleu["bleu4"],
        "bleu1":       bleu["bleu1"],
        "bleu2":       bleu["bleu2"],
        "bleu3":       bleu["bleu3"],
        "chrf":        chrf["chrf"],
        "rouge1":      rouge["rouge1"],
        "rouge2":      rouge["rouge2"],
        "rougeL":      rouge["rougeL"],
        # Language
        "hindi_pct":   lang["hindi_pct"],
        "mismatch_pct":lang["mismatch_pct"],
        "english_pct": lang["english_pct"],
        # Error categories
        "correct_pct": round(error_counts.get("correct", 0)           / total * 100, 2),
        "lang_mismatch_pct":  round(error_counts.get("language_mismatch", 0) / total * 100, 2),
        "topic_drift_pct":    round(error_counts.get("topic_drift",       0) / total * 100, 2),
        "hallucination_pct":  round(error_counts.get("number_hallucination", 0) / total * 100, 2),
        "repetition_pct":     round(error_counts.get("repetition_loop",   0) / total * 100, 2),
        "too_short_pct":      round(error_counts.get("too_short",         0) / total * 100, 2),
        "empty_pct":          round(error_counts.get("empty_response",    0) / total * 100, 2),
        # Length
        "avg_pred_len": round(np.mean(pred_lens), 1),
        "avg_ref_len":  round(np.mean(ref_lens),  1),
        # Latency
        "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
        # Per-intent
        "per_intent": per_intent,
        # Raw errors for analysis
        "_errors": errors,
        "_records": records,
    }


# ── Print master comparison table ─────────────────────────────────────
def print_master_table(all_metrics: dict):
    sep   = "=" * 80
    names = ["zero_shot", "one_shot", "few_shot",
             "prompt_engineered", "finetuned_rag"]
    labels = {
        "zero_shot":         "Zero-shot",
        "one_shot":          "One-shot",
        "few_shot":          "Few-shot (5ex)",
        "prompt_engineered": "Prompt-eng.",
        "finetuned_rag":     "Fine-tuned+RAG",
    }

    def col(name, key, fmt=".2f"):
        m = all_metrics.get(name, {})
        v = m.get(key, 0)
        return f"{v:{fmt}}"

    print(f"\n{sep}")
    print("  KisanMitra AI — Complete Baseline Comparison")
    print(sep)

    # Header
    print(f"\n  {'Metric':<22}", end="")
    for n in names:
        print(f"{labels[n]:>15}", end="")
    print()
    print(f"  {'─'*97}")

    # ── Quantitative Metrics ──
    print(f"\n  ── Quantitative Metrics (↑ higher is better) ──────────────────────────────")

    rows = [
        ("BLEU-4",       "bleu4"),
        ("BLEU-1",       "bleu1"),
        ("BLEU-2",       "bleu2"),
        ("BLEU-3",       "bleu3"),
        ("chrF",         "chrf"),
        ("ROUGE-1",      "rouge1"),
        ("ROUGE-2",      "rouge2"),
        ("ROUGE-L",      "rougeL"),
    ]

    for label, key in rows:
        vals = [all_metrics.get(n, {}).get(key, 0) for n in names]
        max_v = max(vals)
        print(f"  {label:<22}", end="")
        for v in vals:
            marker = " ★" if v == max_v else "  "
            print(f"{v:>13.2f}{marker}", end="")
        print()

    # ── Language & Quality ──
    print(f"\n  ── Language & Quality (Hindi% ↑, Mismatch% ↓) ─────────────────────────────")
    lang_rows = [
        ("Hindi output %",   "hindi_pct"),
        ("Lang mismatch %",  "mismatch_pct"),
        ("English output %", "english_pct"),
        ("Correct %",        "correct_pct"),
        ("Avg pred length",  "avg_pred_len"),
        ("Avg latency ms",   "avg_latency_ms"),
    ]
    for label, key in lang_rows:
        vals = [all_metrics.get(n, {}).get(key, 0) for n in names]
        print(f"  {label:<22}", end="")
        for v in vals:
            print(f"{v:>15.1f}", end="")
        print()

    # ── Error Categories ──
    print(f"\n  ── Error Categories (all ↓ lower is better) ─────────────────────────────")
    err_rows = [
        ("Correct %",          "correct_pct"),
        ("Topic drift %",      "topic_drift_pct"),
        ("Hallucination %",    "hallucination_pct"),
        ("Language mismatch%", "lang_mismatch_pct"),
        ("Repetition %",       "repetition_pct"),
        ("Too short %",        "too_short_pct"),
        ("Empty %",            "empty_pct"),
    ]
    for label, key in err_rows:
        vals = [all_metrics.get(n, {}).get(key, 0) for n in names]
        print(f"  {label:<22}", end="")
        for v in vals:
            print(f"{v:>15.1f}", end="")
        print()

    print(f"\n  ★ = best score in row")
    print(sep)


# ── Per-intent table ──────────────────────────────────────────────────
def print_intent_table(all_metrics: dict):
    sep    = "=" * 80
    names  = ["zero_shot", "one_shot", "few_shot",
              "prompt_engineered", "finetuned_rag"]
    labels = {
        "zero_shot":         "Zero",
        "one_shot":          "One",
        "few_shot":          "Few",
        "prompt_engineered": "PE",
        "finetuned_rag":     "FT+RAG",
    }

    # Collect all intents
    all_intents = set()
    for m in all_metrics.values():
        all_intents.update(m.get("per_intent", {}).keys())
    all_intents = sorted(all_intents)

    print(f"\n{sep}")
    print("  Per-Intent Correct % — All Models")
    print(sep)
    print(f"  {'Intent':<25}", end="")
    for n in names:
        print(f"{labels[n]:>12}", end="")
    print()
    print(f"  {'─'*85}")

    for intent in all_intents:
        vals = []
        for n in names:
            pct = all_metrics.get(n, {}).get("per_intent", {})\
                             .get(intent, {}).get("correct_pct", 0)
            count = all_metrics.get(n, {}).get("per_intent", {})\
                               .get(intent, {}).get("count", 0)
            vals.append((pct, count))

        max_pct = max(v[0] for v in vals) if vals else 0
        print(f"  {intent:<25}", end="")
        for pct, count in vals:
            marker = "★" if pct == max_pct else " "
            print(f"{pct:>10.1f}%{marker}", end="")
        print()

    print(sep)


# ── Improvement delta table ───────────────────────────────────────────
def print_improvement_table(all_metrics: dict):
    sep = "=" * 80
    ft  = all_metrics.get("finetuned_rag", {})

    print(f"\n{sep}")
    print("  Fine-tuned+RAG vs Each Baseline — Improvement Delta")
    print(sep)
    print(f"  {'Metric':<22} {'vs Zero':>10} {'vs One':>10} "
          f"{'vs Few':>10} {'vs PE':>10}")
    print(f"  {'─'*62}")

    baselines = ["zero_shot", "one_shot", "few_shot", "prompt_engineered"]
    metrics   = [
        ("BLEU-4",    "bleu4",      True),
        ("chrF",      "chrf",       True),
        ("ROUGE-1",   "rouge1",     True),
        ("ROUGE-L",   "rougeL",     True),
        ("Hindi %",   "hindi_pct",  True),
        ("Correct %", "correct_pct",True),
        ("Mismatch%", "mismatch_pct",False),
    ]

    for label, key, higher_better in metrics:
        ft_val = ft.get(key, 0)
        print(f"  {label:<22}", end="")
        for b_name in baselines:
            b_val = all_metrics.get(b_name, {}).get(key, 0)
            delta = ft_val - b_val
            arrow = "↑" if (delta > 0 and higher_better) or \
                           (delta < 0 and not higher_better) else "↓"
            print(f"  {arrow}{abs(delta):>7.2f}", end="")
        print()

    print(f"\n  ↑ = Fine-tuned+RAG is better than baseline")
    print(f"  ↓ = Fine-tuned+RAG is worse (rare)")
    print(sep)


# ── Save all metrics to JSON ──────────────────────────────────────────
def save_all_metrics(all_metrics: dict):
    # Remove non-serializable fields before saving
    clean = {}
    for name, m in all_metrics.items():
        clean[name] = {
            k: v for k, v in m.items()
            if not k.startswith("_")
        }

    out_path = RESULTS_DIR / "all_metrics_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    logger.success(f"All metrics saved → {out_path}")


# ── Save per-record predictions with metrics ──────────────────────────
def save_annotated_predictions(all_metrics: dict):
    """Save each model's predictions with their error labels."""
    for name, m in all_metrics.items():
        records = m.get("_records", [])
        errors  = m.get("_errors",  [])
        if not records:
            continue

        out_path = RESULTS_DIR / f"annotated_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record, error in zip(records, errors):
                annotated = {**record, "error_category": error}
                # Remove heavy fields
                annotated.pop("_records", None)
                annotated.pop("_errors",  None)
                f.write(json.dumps(annotated, ensure_ascii=False) + "\n")

        logger.success(f"Annotated predictions → {out_path}")


# ── Save CSV summary for easy sharing ────────────────────────────────
def save_csv_summary(all_metrics: dict):
    import csv
    names  = ["zero_shot", "one_shot", "few_shot",
              "prompt_engineered", "finetuned_rag"]
    keys   = [
        "bleu4", "bleu1", "chrf", "rouge1", "rouge2", "rougeL",
        "hindi_pct", "mismatch_pct", "correct_pct",
        "topic_drift_pct", "hallucination_pct",
        "avg_pred_len", "avg_latency_ms",
    ]

    out_path = RESULTS_DIR / "metrics_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + names)
        for key in keys:
            row = [key] + [
                all_metrics.get(n, {}).get(key, 0)
                for n in names
            ]
            writer.writerow(row)

    logger.success(f"CSV summary → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("KisanMitra AI — Baseline Metrics Computation")
    logger.info("=" * 80)

    all_metrics = {}

    # ── Load all 4 baselines ──
    baseline_names = [
        "zero_shot",
        "one_shot",
        "few_shot",
        "prompt_engineered",
    ]

    for name in baseline_names:
        path = BASELINES_DIR / f"baseline_{name}.jsonl"
        if not path.exists():
            logger.warning(f"Missing: {path} — skipping")
            continue

        records = load_jsonl(str(path))
        logger.info(f"Loaded {len(records)} records for: {name}")

        if records:
            metrics = compute_all_metrics(records, name)
            all_metrics[name] = metrics
            logger.success(
                f"{name}: BLEU4={metrics['bleu4']:.2f} "
                f"chrF={metrics['chrf']:.2f} "
                f"ROUGE1={metrics['rouge1']:.2f} "
                f"Hindi%={metrics['hindi_pct']:.1f}%"
            )

    # ── Load fine-tuned + RAG results ──
    if FINETUNED_PATH.exists():
        ft_records = load_jsonl(str(FINETUNED_PATH))

        # Align with one baseline to get common queries
        if "zero_shot" in all_metrics:
            zs_records = all_metrics["zero_shot"]["_records"]
            ft_aligned, _ = align_records(ft_records, ft_records)

            # Use full fine-tuned records directly
            ft_metrics = compute_all_metrics(ft_records, "finetuned_rag")
            all_metrics["finetuned_rag"] = ft_metrics
            logger.success(
                f"finetuned_rag: BLEU4={ft_metrics['bleu4']:.2f} "
                f"chrF={ft_metrics['chrf']:.2f} "
                f"ROUGE1={ft_metrics['rouge1']:.2f} "
                f"Hindi%={ft_metrics['hindi_pct']:.1f}%"
            )
    else:
        logger.warning(
            f"Fine-tuned results not found at {FINETUNED_PATH}\n"
            "Run evaluate_rag_pipeline.py first."
        )

    if not all_metrics:
        logger.error("No results to compare. Run run_baselines.py first.")
        sys.exit(1)

    # ── Print tables ──
    print_master_table(all_metrics)
    print_intent_table(all_metrics)
    print_improvement_table(all_metrics)

    # ── Save everything ──
    save_all_metrics(all_metrics)
    save_annotated_predictions(all_metrics)
    save_csv_summary(all_metrics)

    # ── Final summary ──
    sep = "=" * 80
    ft  = all_metrics.get("finetuned_rag", {})
    zs  = all_metrics.get("zero_shot",     {})

    print(f"\n{sep}")
    print("  Key Findings")
    print(sep)

    if ft and zs:
        bleu_improvement  = ft.get("bleu4",    0) - zs.get("bleu4",    0)
        rouge_improvement = ft.get("rouge1",   0) - zs.get("rouge1",   0)
        hindi_improvement = ft.get("hindi_pct",0) - zs.get("hindi_pct",0)
        mismatch_reduction= zs.get("mismatch_pct",0) - ft.get("mismatch_pct",0)

        print(f"  Fine-tuned+RAG vs Zero-shot:")
        print(f"    BLEU-4 improvement  : +{bleu_improvement:.2f}")
        print(f"    ROUGE-1 improvement : +{rouge_improvement:.2f}")
        print(f"    Hindi output gain   : +{hindi_improvement:.1f}%")
        print(f"    Mismatch reduction  : -{mismatch_reduction:.1f}%")

    print(f"\n  Files saved:")
    print(f"    {RESULTS_DIR}/all_metrics_comparison.json")
    print(f"    {RESULTS_DIR}/metrics_comparison.csv")
    print(f"    {RESULTS_DIR}/annotated_*.jsonl  (5 files)")
    print(sep)

    logger.success("All metrics computed and saved.")
    logger.success("Next: run scripts/baselines/qualitative_analysis.py")