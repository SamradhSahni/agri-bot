import json
import os
import re
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, ".")

from utils.config_loader import CONFIG, load_config

log_path = Path("logs/clean_dataset.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

CONFIG = load_config()


# ── Load JSONL ───────────────────────────────────────────────────────
def load_jsonl(filepath: str) -> pd.DataFrame:
    records = []
    filepath = Path(filepath)
    logger.info(f"Loading: {filepath.name} ({filepath.stat().st_size/1024/1024:.1f} MB)")
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Reading lines"):
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            logger.success(f"Loaded {len(records):,} records with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            records = []
            continue
    return pd.DataFrame(records)


# ── Step 1: Drop rows with missing required fields ───────────────────
def drop_missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    required = ["query", "answer", "crop", "state"]
    df = df.dropna(subset=required)

    # Also drop where query or answer is empty string
    df = df[df["query"].astype(str).str.strip().str.len() > 0]
    df = df[df["answer"].astype(str).str.strip().str.len() > 0]

    dropped = before - len(df)
    logger.info(f"[Step 1] Drop missing fields: {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 2: Filter to target Hindi-belt states ───────────────────────
def filter_states(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    target_states = [s.upper() for s in CONFIG["target_states"]]
    df = df[df["state"].astype(str).str.upper().str.strip().isin(target_states)]
    dropped = before - len(df)
    logger.info(f"[Step 2] Filter states: {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 3: Drop noise/junk queries ─────────────────────────────────
def drop_noise_queries(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    noise_patterns = CONFIG["noise_query_patterns"]

    def is_noise(query: str) -> bool:
        query_lower = str(query).lower().strip()
        for pattern in noise_patterns:
            if pattern.lower() in query_lower:
                return True
        return False

    mask = df["query"].apply(is_noise)
    df = df[~mask]
    dropped = before - len(df)
    logger.info(f"[Step 3] Drop noise queries: {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 4: Apply query length filter ───────────────────────────────
def filter_query_length(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    min_q = CONFIG["query_length"]["min_chars"]
    max_q = CONFIG["query_length"]["max_chars"]

    df["_query_len"] = df["query"].astype(str).apply(len)
    df = df[(df["_query_len"] >= min_q) & (df["_query_len"] <= max_q)]
    df = df.drop(columns=["_query_len"])

    dropped = before - len(df)
    logger.info(f"[Step 4] Query length filter ({min_q}–{max_q} chars): {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 5: Apply answer length filter ──────────────────────────────
def filter_answer_length(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    min_a = CONFIG["answer_length"]["min_chars"]
    max_a = CONFIG["answer_length"]["max_chars"]

    df["_answer_len"] = df["answer"].astype(str).apply(len)
    df = df[(df["_answer_len"] >= min_a) & (df["_answer_len"] <= max_a)]
    df = df.drop(columns=["_answer_len"])

    dropped = before - len(df)
    logger.info(f"[Step 5] Answer length filter ({min_a}–{max_a} chars): {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 6: Validate answers are in Hindi (Devanagari) ──────────────
def is_hindi(text: str, threshold: float = None) -> bool:
    """
    Returns True if the text has enough Devanagari characters.
    """
    if threshold is None:
        threshold = CONFIG["hindi_detection"]["min_hindi_ratio"]
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) == 0:
        return False
    devanagari = sum(
        1 for c in text
        if '\u0900' <= c <= '\u097F'
    )
    ratio = devanagari / len(text)
    return ratio >= threshold


def validate_hindi_answers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    tqdm.pandas(desc="Validating Hindi answers")
    df["_is_hindi_answer"] = df["answer"].astype(str).progress_apply(is_hindi)
    df = df[df["_is_hindi_answer"]]
    df = df.drop(columns=["_is_hindi_answer"])
    dropped = before - len(df)
    logger.info(f"[Step 6] Hindi answer validation: {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 7: Standardise columns & clean text ─────────────────────────
def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalise state names to UPPER
    - Strip leading/trailing whitespace from all string fields
    - Collapse multiple spaces/newlines in answers
    - Ensure crop is lowercase for consistency
    """
    before = len(df)

    df["state"]  = df["state"].astype(str).str.upper().str.strip()
    df["crop"]   = df["crop"].astype(str).str.lower().str.strip()
    df["query"]  = df["query"].astype(str).str.strip()
    df["answer"] = (
        df["answer"]
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)   # collapse whitespace
        .str.replace(r'\n+', ' ', regex=True)   # remove newlines
    )

    # Fix JHARKAND → JHARKHAND for consistency in output
    df["state"] = df["state"].replace({"JHARKAND": "JHARKHAND"})

    # Drop any rows that became empty after stripping
    df = df[df["query"].str.len() > 0]
    df = df[df["answer"].str.len() > 0]

    dropped = before - len(df)
    logger.info(f"[Step 7] Standardise & clean text: {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Step 8: Tag intent on every row ─────────────────────────────────
def detect_intent(query: str, answer: str) -> str:
    """
    Try query first (English), then answer (Hindi).
    Return first matched intent or 'unknown'.
    """
    intents = CONFIG["intents"]
    query_lower  = str(query).lower()
    answer_lower = str(answer).lower()

    for intent_name, kw_banks in intents.items():
        english_kws = kw_banks.get("english", [])
        hindi_kws   = kw_banks.get("hindi", [])

        # Check query (English)
        for kw in english_kws:
            if kw.lower() in query_lower:
                return intent_name

        # Check answer (Hindi)
        for kw in hindi_kws:
            if kw in answer_lower:
                return intent_name

    return "unknown"


def tag_intents(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    tqdm.pandas(desc="Tagging intents")
    df["intent"] = df.progress_apply(
        lambda row: detect_intent(row["query"], row["answer"]),
        axis=1
    )

    # Print intent distribution
    dist = df["intent"].value_counts()
    logger.info("Intent distribution after tagging:")
    for intent, count in dist.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {intent:<25} {count:>8,}  ({pct:.1f}%)")

    logger.info(f"[Step 8] Intent tagging complete: {len(df):,} records")
    return df


# ── Step 9: Drop 'unknown' intent rows (optional but recommended) ────
def drop_unknown_intents(df: pd.DataFrame, keep_unknown: bool = False) -> pd.DataFrame:
    if keep_unknown:
        logger.info("[Step 9] Keeping unknown intent rows as-is")
        return df
    before = len(df)
    df = df[df["intent"] != "unknown"]
    dropped = before - len(df)
    logger.info(f"[Step 9] Drop unknown intents: {before:,} → {len(df):,} (dropped {dropped:,})")
    return df


# ── Print cleaning summary ───────────────────────────────────────────
def print_cleaning_report(original_count: int, df: pd.DataFrame):
    sep = "=" * 55
    print(f"\n{sep}")
    print("  Cleaning Pipeline — Final Report")
    print(sep)
    print(f"  Original records  : {original_count:,}")
    print(f"  Clean records     : {len(df):,}")
    print(f"  Dropped total     : {original_count - len(df):,}")
    print(f"  Retention rate    : {(len(df)/original_count)*100:.1f}%")

    print(f"\n  State distribution:")
    for state, count in df["state"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"    {state:<25} {count:>8,}  ({pct:.1f}%)")

    print(f"\n  Intent distribution:")
    for intent, count in df["intent"].value_counts().items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"    {intent:<25} {count:>8,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Answer length stats (post-clean):")
    ans_lens = df["answer"].str.len()
    print(f"    Min    : {ans_lens.min()}")
    print(f"    Max    : {ans_lens.max()}")
    print(f"    Mean   : {ans_lens.mean():.1f}")
    print(f"    Median : {ans_lens.median():.1f}")

    print(f"\n{sep}")


# ── Save cleaned JSONL ───────────────────────────────────────────────
def save_jsonl(df: pd.DataFrame, filepath: str):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Keep only needed columns
    cols = ["query", "answer", "crop", "state", "intent", "language", "source", "section"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving"):
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    logger.success(f"Saved {len(df):,} records to: {filepath}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_path       = os.getenv("RAW_DATA_PATH",       "./data/raw/kcc_dataset.jsonl")
    filtered_path  = os.getenv("PROCESSED_DATA_PATH", "./data/processed/clean_dataset.jsonl")

    logger.info("=" * 55)
    logger.info("KisanMitra AI — Data Cleaning Pipeline (Task 4a)")
    logger.info("=" * 55)

    # Load
    df = load_jsonl(raw_path)
    original_count = len(df)

    # Run all cleaning steps in order
    df = drop_missing_fields(df)     # Step 1
    df = filter_states(df)           # Step 2
    df = drop_noise_queries(df)      # Step 3
    df = filter_query_length(df)     # Step 4
    df = filter_answer_length(df)    # Step 5
    df = validate_hindi_answers(df)  # Step 6
    df = standardise(df)             # Step 7
    df = tag_intents(df)             # Step 8
    df = drop_unknown_intents(df,    # Step 9
            keep_unknown=False)

    # Report
    print_cleaning_report(original_count, df)

    # Save
    save_jsonl(df, filtered_path)

    logger.success("Task 4a complete — clean_dataset.jsonl saved.")