from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("⏳ Loading tokenizer and model (downloads ~3 GB on first run)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically uses GPU if available, otherwise CPU
)

print("✅ Model and tokenizer loaded successfully!")