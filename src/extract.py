from transformers import pipeline
import json, re
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = Path(__file__).parent.parent

print("⏳ Initializing extraction pipeline...")
generator = pipeline("text-generation", model=MODEL_NAME, torch_dtype="auto", device_map="auto", return_full_text=False)

def extract_medical_data(raw_text: str) -> dict:
    prompt = f"""Du bist ein medizinischer Datenassistent. Extrahiere Informationen aus dem folgenden deutschen Arztbrief und antworte AUSSCHLIESSLICH mit einem gültigen JSON-Objekt. Keine Einleitungen, keine Erklärungen, kein Markdown.

Erwartetes JSON-Format (alle Felder müssen vorhanden sein, leere Listen [] wenn nicht zutreffend):
{{
  "patient_name": "string",
  "dob": "DD.MM.YYYY",
  "insurance_id": "string",
  "visit_date": "DD.MM.YYYY",
  "diagnoses": ["string"],
  "symptoms": ["string"],
  "medications": [{{"name": "string", "dosage": "string", "frequency": "string"}}],
  "allergies": ["string"],
  "recommendations": ["string"]
}}

WICHTIG: Antworte NUR auf Deutsch. Formatiere Häufigkeiten konsequent als "1x täglich", "2x wöchentlich" etc.
Arztbrief:
{raw_text}
"""

    messages = [{"role": "user", "content": prompt}]
    try:
        output = generator(messages, max_new_tokens=400, do_sample=False, temperature=0.1)[0]["generated_text"]
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
    return {"error": "Extraction failed"}

if __name__ == "__main__":
    letter_path = BASE_DIR / "data" / "letter_01.txt"
    print("\n🔍 Running extraction...")
    with open(letter_path, "r", encoding="utf-8") as f:
        letter = f.read()
    
    result = extract_medical_data(letter)
    
    out_path = BASE_DIR / "data" / "extracted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON saved to: {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))