import os
import sys
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/setup_model.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME      = os.getenv("BASE_MODEL_NAME", "google/mt5-base")
CHECKPOINT_DIR  = os.getenv("CHECKPOINT_DIR",  "./model/checkpoints")
FINAL_MODEL_DIR = os.getenv("FINETUNED_MODEL_PATH", "./model/final")


# ── Check VRAM before loading ─────────────────────────────────────────
def check_gpu():
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected — will run on CPU (extremely slow)")
        return False, 0

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU   : {gpu_name}")
    logger.info(f"VRAM  : {vram_gb:.1f} GB")

    if vram_gb < 4:
        logger.error("Less than 4GB VRAM — QLoRA may fail. Reduce batch size.")
    elif vram_gb < 8:
        logger.warning("6GB VRAM — QLoRA will work, keep batch_size=4")
    else:
        logger.success("8GB+ VRAM — comfortable for QLoRA fine-tuning")

    return True, vram_gb


# ── Load tokenizer ────────────────────────────────────────────────────
def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    # mT5 uses sentencepiece — verify it works
    test_text = "किसान भाई गेहूं में सिंचाई कब करें?"
    tokens    = tokenizer(test_text, return_tensors="pt")
    decoded   = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    logger.success(f"Tokenizer loaded — vocab size: {tokenizer.vocab_size:,}")
    logger.info(f"  Test encode/decode: '{test_text}'")
    logger.info(f"  Token count       : {tokens['input_ids'].shape[1]}")
    logger.info(f"  Decoded back      : '{decoded}'")

    return tokenizer


# ── Load model with 4-bit NF4 QLoRA quantization ─────────────────────
def load_quantized_model(model_name: str):
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

    logger.info(f"Loading model with 4-bit NF4 quantization: {model_name}")
    logger.info("First run will download ~580MB — please wait...")

    # BitsAndBytes 4-bit NF4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # NF4 quantization
        bnb_4bit_use_double_quant=True,          # nested quantization for extra memory savings
        bnb_4bit_compute_dtype=torch.float16,    # compute in fp16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                       # auto-place on GPU
        trust_remote_code=True,
    )

    # Print model memory footprint
    mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
    mem_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    mem_total_mb = (mem_params + mem_buffers) / 1024**2

    logger.success(f"Model loaded — memory footprint: {mem_total_mb:.0f} MB")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


# ── Print model architecture ──────────────────────────────────────────
def inspect_model_layers(model):
    """
    Print layer names so we can confirm which ones LoRA will target.
    """
    logger.info("\nModel layer names (relevant projection layers):")
    seen = set()
    for name, module in model.named_modules():
        # Extract the last part of the layer name
        parts = name.split(".")
        layer_type = parts[-1] if parts else name

        if layer_type in ["q", "k", "v", "o", "wi_0", "wi_1", "wo"] and layer_type not in seen:
            logger.info(f"  Found target layer → '{layer_type}' in '{name}'")
            seen.add(layer_type)

    if not seen:
        logger.warning("No target layers found by name — listing all linear layers:")
        for name, module in model.named_modules():
            if "Linear" in type(module).__name__:
                logger.info(f"  {name}: {type(module).__name__}")


# ── Verify full setup ─────────────────────────────────────────────────
def verify_setup(model, tokenizer):
    """
    Run a quick forward pass to confirm model loads and runs.
    """
    logger.info("\nRunning verification forward pass...")

    test_input  = "निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान का प्रश्न: गेहूं में सिंचाई कब करें? उत्तर:"
    test_target = "गेहूं में बुवाई के 20-25 दिन बाद पहली सिंचाई करें।"

    inputs  = tokenizer(test_input,  return_tensors="pt", truncation=True, max_length=256)
    targets = tokenizer(test_target, return_tensors="pt", truncation=True, max_length=128)

    # Move to same device as model
    device  = next(model.parameters()).device
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels         = targets["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    logger.success(f"Forward pass OK — loss: {outputs.loss.item():.4f}")
    logger.info(f"Logits shape: {outputs.logits.shape}")
    return True


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KisanMitra AI — Model Setup Verification (Task 7a)")
    logger.info("=" * 60)

    # GPU check
    has_gpu, vram = check_gpu()

    # Load tokenizer
    tokenizer = load_tokenizer(MODEL_NAME)

    # Load quantized model
    model = load_quantized_model(MODEL_NAME)

    # Inspect layers
    inspect_model_layers(model)

    # Verify forward pass
    verify_setup(model, tokenizer)

    sep = "=" * 60
    logger.success(f"\n{sep}")
    logger.success("Task 7a complete — model loads and runs correctly.")
    logger.success(f"Model  : {MODEL_NAME}")
    logger.success(f"VRAM   : {vram:.1f} GB available")
    logger.success("Ready for LoRA adapter config in Task 7b.")
    logger.success(sep)