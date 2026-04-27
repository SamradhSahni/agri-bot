import sys
import torch
from pathlib import Path
from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "./IndicTrans2")

def test_translation():
    logger.info("Loading IndicTrans2 model...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {vram:.1f} GB")

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor

        # Use 200M model — safe for 6GB VRAM
        # If you want higher quality use: ai4bharat/indictrans2-en-indic-1B
        MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info("This will download ~800MB on first run...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        ip = IndicProcessor(inference=True)

        # Test sentences — real KCC-style queries
        test_sentences = [
            "Farmer asked query on Weather",
            "Query about protection of fall army worm in maize",
            "Chief Minister New and Renewable Solar Pump Scheme",
            "Query about Kisan Credit Card",
            "ASKED ABOUT TO NUTRIENT MANAGEMENT IN WHEAT",
            "Information regarding how to improve growth in Paddy",
            "Farmer asked for Isabgol farming",
            "Query about pest management in cotton crop",
        ]

        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"

        logger.info(f"\nTranslating {len(test_sentences)} test sentences...")

        # Preprocess
        batch = ip.preprocess_batch(
            test_sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess
        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        # Print results
        sep = "=" * 60
        print(f"\n{sep}")
        print("  IndicTrans2 — Test Translation Results")
        print(sep)
        for original, translated in zip(test_sentences, translations):
            print(f"\n  EN: {original}")
            print(f"  HI: {translated}")
        print(f"\n{sep}")

        logger.success("IndicTrans2 is working correctly!")
        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure IndicTrans2 repo is cloned and IndicTransToolkit is installed")
        return False
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        raise


if __name__ == "__main__":
    test_translation()