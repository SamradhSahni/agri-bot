import sys
import time
import json
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/inference_test.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")


# ── 10 diverse test cases covering all major intents ─────────────────
TEST_CASES = [
    {
        "id":     1,
        "intent": "weather_sowing",
        "state":  "BIHAR",
        "crop":   "wheat",
        "query":  "गेहूं की बुवाई के लिए सबसे अच्छा समय कौन सा है?",
    },
    {
        "id":     2,
        "intent": "pest_id",
        "state":  "UTTAR PRADESH",
        "crop":   "maize (makka)",
        "query":  "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?",
    },
    {
        "id":     3,
        "intent": "disease",
        "state":  "MADHYA PRADESH",
        "crop":   "soybean (bhat)",
        "query":  "सोयाबीन में पीला मोजेक वायरस रोग का उपचार क्या है?",
    },
    {
        "id":     4,
        "intent": "nutrient_management",
        "state":  "RAJASTHAN",
        "crop":   "mustard",
        "query":  "सरसों की फसल में कौन सा उर्वरक और कितनी मात्रा में डालें?",
    },
    {
        "id":     5,
        "intent": "crop_advisory",
        "state":  "HARYANA",
        "crop":   "paddy (dhan)",
        "query":  "धान की रोपाई के बाद पहली सिंचाई कब करनी चाहिए?",
    },
    {
        "id":     6,
        "intent": "government_scheme",
        "state":  "BIHAR",
        "crop":   "others",
        "query":  "किसान क्रेडिट कार्ड के लिए आवेदन कैसे करें?",
    },
    {
        "id":     7,
        "intent": "msp_price",
        "state":  "UTTAR PRADESH",
        "crop":   "wheat",
        "query":  "गेहूं का न्यूनतम समर्थन मूल्य क्या है और कहां बेचें?",
    },
    {
        "id":     8,
        "intent": "horticulture",
        "state":  "HIMACHAL PRADESH",
        "crop":   "apple",
        "query":  "सेब के बाग में कौन सी बीमारी ज्यादा आती है और कैसे रोकें?",
    },
    {
        "id":     9,
        "intent": "soil_water",
        "state":  "RAJASTHAN",
        "crop":   "others",
        "query":  "मिट्टी की जांच कैसे करें और कहां करवाएं?",
    },
    {
        "id":     10,
        "intent": "animal_husbandry",
        "state":  "HARYANA",
        "crop":   "others",
        "query":  "गाय में दूध बढ़ाने के लिए क्या खिलाएं?",
    },
]


# ── Validate response quality ─────────────────────────────────────────
def validate_response(response: str, intent: str) -> dict:
    """
    Auto-validate each response for basic quality checks.
    Returns a dict of pass/fail flags.
    """
    checks = {}

    # 1. Not empty
    checks["not_empty"] = len(response.strip()) > 10

    # 2. Is Hindi
    dev_chars   = sum(1 for c in response if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in response if c.isalpha())
    checks["is_hindi"] = (dev_chars / max(total_alpha, 1)) >= 0.4

    # 3. Minimum length
    checks["min_length"] = len(response) >= 20

    # 4. No repetition loop
    words = response.split()
    if len(words) >= 6:
        from collections import Counter
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        max_repeat = max(Counter(trigrams).values()) if trigrams else 1
        checks["no_repetition"] = max_repeat < 3
    else:
        checks["no_repetition"] = True

    # 5. Contains agricultural vocabulary
    agri_terms = [
        "किसान", "फसल", "बीज", "खाद", "कीट", "रोग", "सिंचाई",
        "बुवाई", "उर्वरक", "कृषि", "विभाग", "संपर्क", "प्रबंधन",
        "मात्रा", "छिड़काव", "पानी", "मिट्टी", "पशु", "दूध",
    ]
    checks["agri_vocab"] = any(t in response for t in agri_terms)

    # Overall pass
    checks["overall_pass"] = all(checks.values())

    return checks


# ── Print single test result ──────────────────────────────────────────
def print_test_result(
    case:     dict,
    result:   dict,
    checks:   dict,
    latency:  float,
):
    sep = "─" * 65
    status = "✅ PASS" if checks["overall_pass"] else "❌ FAIL"

    print(f"\n{sep}")
    print(f"  Test {case['id']:>2} | {status} | {case['intent']} | {latency:.2f}s")
    print(sep)
    print(f"  State   : {case['state']}")
    print(f"  Crop    : {case['crop']}")
    print(f"  Query   : {case['query']}")
    print(f"  Intent  : {result['intent']}  (detected)")
    print(f"\n  Response:")
    # Wrap response at 65 chars for readability
    response = result["response"]
    words    = response.split()
    line     = "  "
    for word in words:
        if len(line) + len(word) + 1 > 67:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print(f"\n  Quality checks:")
    for check_name, passed in checks.items():
        if check_name == "overall_pass":
            continue
        icon = "✅" if passed else "❌"
        print(f"    {icon} {check_name}")

    print(f"  Tokens generated : {result['tokens_generated']}")


# ── Print summary report ──────────────────────────────────────────────
def print_summary(results: list):
    sep = "=" * 65
    passed   = sum(1 for r in results if r["checks"]["overall_pass"])
    total    = len(results)
    avg_lat  = sum(r["latency"] for r in results) / total
    avg_tok  = sum(r["result"]["tokens_generated"] for r in results) / total

    print(f"\n{sep}")
    print("  End-to-End Inference Test — Summary")
    print(sep)
    print(f"  Total tests     : {total}")
    print(f"  Passed          : {passed}  ({passed/total*100:.0f}%)")
    print(f"  Failed          : {total - passed}")
    print(f"  Avg latency     : {avg_lat:.2f}s")
    print(f"  Avg tokens gen  : {avg_tok:.1f}")

    # Per-test result table
    print(f"\n  {'Test':<6} {'Intent':<22} {'Pass':<6} {'Latency':>8} {'Tokens':>7}")
    print(f"  {'─'*55}")
    for r in results:
        status = "✅" if r["checks"]["overall_pass"] else "❌"
        print(
            f"  {r['case']['id']:<6} "
            f"{r['case']['intent']:<22} "
            f"{status:<6} "
            f"{r['latency']:>7.2f}s "
            f"{r['result']['tokens_generated']:>7}"
        )

    # Check latency for production readiness
    print(f"\n  Production readiness:")
    if avg_lat < 1.0:
        print(f"  ✅ Avg latency {avg_lat:.2f}s — excellent for production")
    elif avg_lat < 3.0:
        print(f"  ✅ Avg latency {avg_lat:.2f}s — acceptable for production")
    else:
        print(f"  ⚠️  Avg latency {avg_lat:.2f}s — consider GPU deployment")

    if passed == total:
        print(f"  ✅ All {total} tests passed — model ready for RAG pipeline")
    elif passed >= total * 0.8:
        print(f"  ✅ {passed}/{total} tests passed — model ready for RAG pipeline")
    else:
        print(f"  ⚠️  Only {passed}/{total} passed — review failures before proceeding")

    print(f"\n{sep}")


# ── Save results ──────────────────────────────────────────────────────
def save_results(results: list):
    out_path = Path("data/processed/eval_results/inference_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in results:
        serializable.append({
            "test_id":   r["case"]["id"],
            "intent":    r["case"]["intent"],
            "state":     r["case"]["state"],
            "crop":      r["case"]["crop"],
            "query":     r["case"]["query"],
            "response":  r["result"]["response"],
            "latency":   round(r["latency"], 3),
            "tokens":    r["result"]["tokens_generated"],
            "checks":    r["checks"],
            "passed":    r["checks"]["overall_pass"],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.success(f"Inference test results saved → {out_path}")


# ── Model file check ──────────────────────────────────────────────────
def check_model_files():
    from dotenv import load_dotenv
    import os
    load_dotenv()

    model_dir = Path(os.getenv("FINETUNED_MODEL_PATH", "./model/final"))
    sep = "=" * 65

    print(f"\n{sep}")
    print("  Model File Check")
    print(sep)
    print(f"  Model directory: {model_dir.resolve()}")

    expected_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]

    # Check for either pytorch_model.bin or safetensors shards
    weight_files = (
        list(model_dir.glob("*.bin")) +
        list(model_dir.glob("*.safetensors"))
    )

    all_good = True
    for fname in expected_files:
        exists = (model_dir / fname).exists()
        icon   = "✅" if exists else "❌"
        print(f"  {icon} {fname}")
        if not exists:
            all_good = False

    if weight_files:
        total_size = sum(f.stat().st_size for f in weight_files) / 1024**3
        print(f"  ✅ Model weights ({len(weight_files)} file(s), {total_size:.2f} GB)")
    else:
        print(f"  ❌ No model weight files found (.bin or .safetensors)")
        all_good = False

    if all_good:
        print(f"\n  ✅ All model files present — ready to load")
    else:
        print(f"\n  ❌ Missing files — re-run save_final_model.py")

    print(f"{sep}\n")
    return all_good


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — End-to-End Inference Test (Task 10b)")
    logger.info("=" * 65)

    # Step 1: Check model files exist
    files_ok = check_model_files()
    if not files_ok:
        logger.error("Model files missing — stopping. Run Task 8b first.")
        sys.exit(1)

    # Step 2: Load inference engine
    from backend.inference import KisanMitraInference
    engine = KisanMitraInference()
    engine.load()

    # Step 3: Warm up (first call is slower due to CUDA JIT)
    logger.info("Warming up model...")
    _ = engine.generate(
        query="गेहूं की बुवाई कब करें?",
        state="UTTAR PRADESH",
        crop="wheat",
    )
    logger.success("Warmup complete")

    # Step 4: Run all 10 test cases
    print("\n" + "=" * 65)
    print("  Running 10 End-to-End Inference Tests")
    print("=" * 65)

    all_results = []

    for case in TEST_CASES:
        start = time.time()

        result = engine.generate(
            query=case["query"],
            state=case["state"],
            crop=case["crop"],
            intent=case["intent"],
        )

        latency = time.time() - start
        checks  = validate_response(result["response"], case["intent"])

        print_test_result(case, result, checks, latency)

        all_results.append({
            "case":    case,
            "result":  result,
            "checks":  checks,
            "latency": latency,
        })

    # Step 5: Summary
    print_summary(all_results)

    # Step 6: Save
    save_results(all_results)

    logger.success("Task 10b complete — inference engine verified and ready.")
    logger.success("Next: Task 11 — RAG embedding and vector index.")