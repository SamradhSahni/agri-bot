import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
sys.path.insert(0, ".")

from scripts.baselines.nvidia_client  import call_nvidia, test_connection
from scripts.baselines.baseline_prompts import (
    build_zero_shot,
    build_one_shot,
    build_few_shot,
    build_prompt_engineered,
)

log_path = Path("logs/baselines.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────
TEST_PATH      = "./data/processed/test.jsonl"
RESULTS_DIR    = Path("./data/processed/eval_results/baselines")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_SAMPLE    = 200   # 200 samples per baseline — 4 baselines = 800 API calls
DELAY_BETWEEN  = 1.5   # seconds between calls to avoid rate limit

# ── Intent to Hindi display ────────────────────────────────────────────
INTENT_HINDI = {
    "weather_sowing":      "मौसम एवं बुवाई",
    "crop_advisory":       "फसल सलाह",
    "pest_id":             "कीट प्रबंधन",
    "disease":             "रोग प्रबंधन",
    "nutrient_management": "पोषक तत्व प्रबंधन",
    "msp_price":           "मूल्य एवं बाजार",
    "government_scheme":   "सरकारी योजना",
    "horticulture":        "बागवानी",
    "soil_water":          "मृदा एवं जल",
    "animal_husbandry":    "पशुपालन",
    "equipment_machinery": "कृषि यंत्र",
}

STATE_HINDI = {
    "UTTAR PRADESH":    "उत्तर प्रदेश",
    "RAJASTHAN":        "राजस्थान",
    "MADHYA PRADESH":   "मध्य प्रदेश",
    "BIHAR":            "बिहार",
    "HARYANA":          "हरियाणा",
    "JHARKHAND":        "झारखंड",
    "UTTARAKHAND":      "उत्तराखंड",
    "CHHATTISGARH":     "छत्तीसगढ़",
    "HIMACHAL PRADESH": "हिमाचल प्रदेश",
}

CROP_HINDI = {
    "wheat":                  "गेहूं",
    "paddy (dhan)":           "धान",
    "maize (makka)":          "मक्का",
    "mustard":                "सरसों",
    "others":                 "अन्य",
    "soybean (bhat)":         "सोयाबीन",
    "cotton (kapas)":         "कपास",
    "sugarcane (noble cane)": "गन्ना",
    "groundnut (pea nut/mung phalli)": "मूंगफली",
    "green gram (moong bean/ moong)":  "मूंग",
    "pearl millet (bajra/bulrush millet/spiked millet)": "बाजरा",
}


# ── Load test records ─────────────────────────────────────────────────
def load_test_records(filepath: str, sample_size: int) -> list:
    import random
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    continue

    # Stratified sample by intent
    random.seed(42)
    intent_groups = defaultdict(list)
    for r in records:
        intent_groups[r.get("intent", "unknown")].append(r)

    sampled = []
    total   = len(records)
    for intent, group in intent_groups.items():
        n = max(1, int((len(group) / total) * sample_size))
        n = min(n, len(group))
        sampled.extend(random.sample(group, n))

    sampled = sampled[:sample_size]
    random.shuffle(sampled)
    logger.info(f"Loaded {len(sampled)} stratified test records")
    return sampled


# ── Extract Hindi query from formatted input_text ─────────────────────
def extract_query(record: dict) -> str:
    input_text = str(record.get("input_text", ""))
    if "किसान का प्रश्न:" in input_text:
        return input_text.split("किसान का प्रश्न:")[-1]\
                         .split("\nउत्तर:")[0].strip()
    return input_text[:200]


# ── Run a single baseline on all records ─────────────────────────────
def run_baseline(
    name:       str,
    records:    list,
    build_fn,   # one of the four builder functions
) -> list:
    """
    Run one baseline (zero/one/few/prompt-engineered) on all records.
    Returns list of result dicts.
    """
    logger.info(f"\nRunning baseline: {name} ({len(records)} records)...")
    results = []

    for i, record in enumerate(tqdm(records, desc=name)):
        query   = extract_query(record)
        state   = str(record.get("state", "UTTAR PRADESH"))
        crop    = str(record.get("crop",  "others"))
        intent  = str(record.get("intent","unknown"))
        target  = str(record.get("target_text", ""))

        # Translate to Hindi for prompt
        state_hi  = STATE_HINDI.get(state.upper(), state)
        crop_hi   = CROP_HINDI.get(crop.lower(), crop)
        intent_hi = INTENT_HINDI.get(intent, "कृषि सलाह")

        # Build messages
        if name == "prompt_engineered":
            messages = build_fn(query, state_hi, crop_hi, intent_hi)
        else:
            messages = build_fn(query, state_hi, crop_hi)

        # Call API
        start      = time.time()
        prediction = call_nvidia(messages, temperature=0.2, max_tokens=200)
        latency    = int((time.time() - start) * 1000)

        results.append({
            "baseline":   name,
            "query":      query,
            "prediction": prediction,
            "reference":  target,
            "intent":     intent,
            "state":      state,
            "crop":       crop,
            "latency_ms": latency,
        })

        # Rate limit protection
        time.sleep(DELAY_BETWEEN)

        # Progress log every 50 records
        if (i + 1) % 50 == 0:
            logger.info(f"  {name}: {i+1}/{len(records)} done")

    logger.success(f"Baseline '{name}' complete: {len(results)} results")
    return results


# ── Save results to JSONL ─────────────────────────────────────────────
def save_results(results: list, name: str):
    path = RESULTS_DIR / f"baseline_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(results)} results → {path}")
    return path


# ── Print sample outputs for each baseline ────────────────────────────
def print_sample_outputs(all_results: dict):
    sep = "=" * 70

    print(f"\n{sep}")
    print("  Sample Outputs — All Baselines on Same 3 Queries")
    print(sep)

    # Get first 3 queries from zero_shot results as anchor
    anchor_queries = [
        r["query"] for r in all_results["zero_shot"][:3]
    ]

    for i, anchor_q in enumerate(anchor_queries, 1):
        print(f"\n  ── Query {i} ──────────────────────────────────────────────")

        # Find this query in all baselines
        for baseline_name, results in all_results.items():
            match = next((r for r in results if r["query"] == anchor_q), None)
            if match:
                pred = match["prediction"][:200]
                print(f"\n  [{baseline_name.upper()}]")
                print(f"  {pred}")

        # Find reference
        ref_match = next(
            (r for r in all_results["zero_shot"] if r["query"] == anchor_q),
            None
        )
        if ref_match:
            print(f"\n  [REFERENCE]")
            print(f"  {ref_match['reference'][:200]}")

        print(f"\n  {'─'*65}")


# ── Print API usage estimate ──────────────────────────────────────────
def print_usage_estimate(n_records: int):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  NVIDIA NIM API Usage Estimate")
    print(sep)
    print(f"  Records per baseline : {n_records}")
    print(f"  Baselines to run     : 4")
    print(f"  Total API calls      : {n_records * 4:,}")
    print(f"  Delay between calls  : {DELAY_BETWEEN}s")
    est_minutes = (n_records * 4 * (DELAY_BETWEEN + 2)) / 60
    print(f"  Estimated time       : ~{est_minutes:.0f} minutes")
    print(f"  Free tier limit      : 1000 calls/day on NVIDIA NIM")
    print(f"\n  [!] With 200 samples x 4 = 800 calls -- within free tier")
    print(f"  [OK] Safe to run all 4 baselines in one session")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("KisanMitra AI — Baseline Evaluation (All 4 Baselines)")
    logger.info("=" * 70)

    # Test API connection first
    if not test_connection():
        logger.error("NVIDIA API not reachable. Check NVIDIA_API_KEY in .env")
        sys.exit(1)

    # Show usage estimate
    print_usage_estimate(EVAL_SAMPLE)

    print("\nPress Enter to start all 4 baselines (Ctrl+C to cancel)...")
    input()

    # Load test records — same set for all baselines
    records = load_test_records(TEST_PATH, sample_size=EVAL_SAMPLE)

    # ── Run all 4 baselines ──
    baseline_configs = [
        ("zero_shot",          build_zero_shot),
        ("one_shot",           build_one_shot),
        ("few_shot",           build_few_shot),
        ("prompt_engineered",  build_prompt_engineered),
    ]

    all_results = {}

    for name, build_fn in baseline_configs:
        results = run_baseline(name, records, build_fn)
        save_results(results, name)
        all_results[name] = results

        # Small pause between baselines
        logger.info("Pausing 5s between baselines...")
        time.sleep(5)

    # Print sample outputs for visual inspection
    print_sample_outputs(all_results)

    # Save combined summary
    summary = {
        "eval_sample_size": EVAL_SAMPLE,
        "baselines_run":    list(all_results.keys()),
        "records_per_baseline": {
            name: len(results)
            for name, results in all_results.items()
        },
        "note": "Run compute_baseline_metrics.py next for BLEU/ROUGE scores"
    }
    with open(RESULTS_DIR / "baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.success("=" * 70)
    logger.success("All 4 baselines complete.")
    logger.success(f"Results saved to: {RESULTS_DIR}")
    logger.success("Next: run scripts/baselines/compute_baseline_metrics.py")
    logger.success("=" * 70)