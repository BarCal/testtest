# test_pipeline.py
from src.pipeline import load_model, process_letter

print("🔧 Testing pipeline outside Streamlit...")
tokenizer, model, device = load_model()

with open("data/letter_01.txt", "r", encoding="utf-8") as f:
    letter = f.read()

result = process_letter(letter, tokenizer, model, device)
print("\n🎯 Result:", result)