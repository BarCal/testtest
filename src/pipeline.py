# src/pipeline.py
import torch, re, json, sqlite3, os, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = Path(__file__).parent.parent
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DB_PATH = BASE_DIR / "db" / "med_data.db"

def load_model():
    print("⏳ Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    print(f"✅ Model loaded on {device.upper()}")
    return tokenizer, model, device

def extract_json(text: str, tokenizer, model, device: str) -> dict:
    print("🔍 Starting extraction...")
    start_time = time.time()
    
    # Clear prompt with explicit instructions - NO example values that could be copied
    prompt = f"""Du bist ein medizinischer Datenassistent. Lies den folgenden deutschen Arztbrief und extrahiere ALLE Informationen als JSON.

REGELN:
- Antworte NUR mit einem gültigen JSON-Objekt, kein Markdown, kein erklärender Text
- Verwende die TATSÄCHLICHEN Werte aus dem Brief, KEINE Platzhalter
- Wenn ein Feld fehlt, verwende null oder []

Extrahiere diese Felder: patient_name, dob, insurance_id, visit_date, diagnoses[], symptoms[], medications[] (mit name/dosage), allergies[], recommendations[]

ARZTBRIEF:
{text}

JSON-Antwort:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Clean GenerationConfig - NO invalid flags
    gen_config = GenerationConfig(
        max_new_tokens=600,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens=100
    )
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_config)
    
    elapsed = time.time() - start_time
    print(f"⏱️ Generation completed in {elapsed:.2f}s")
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove chat template artifacts and markdown
    output = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', output, flags=re.DOTALL)
    output = re.sub(r'```json\s*', '', output, flags=re.IGNORECASE)
    output = re.sub(r'```\s*', '', output)
    output = output.strip()
    
    print(f"📝 Raw output preview: {output[:300]}...")
    
    # Find JSON: look for first { and match balanced braces
    start_idx = output.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in output")
    
    # Simple brace counter to find matching closing brace
    brace_count = 0
    end_idx = -1
    for i, char in enumerate(output[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        raise ValueError("Could not find complete JSON object")
    
    json_str = output[start_idx:end_idx]
    print(f"✅ Extracted JSON: {json_str[:200]}...")
    
    return json.loads(json_str)

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY, name TEXT, dob TEXT, insurance_id TEXT UNIQUE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, patient_id INTEGER, visit_date TEXT, raw_text TEXT, FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS medical_entities (id INTEGER PRIMARY KEY, document_id INTEGER, entity_type TEXT, entity_value TEXT, FOREIGN KEY(document_id) REFERENCES documents(id))''')
    conn.commit()
    conn.close()

def save_to_db(data: dict, raw_text: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO patients (name, dob, insurance_id) VALUES (?, ?, ?)", 
              (data["patient_name"], data["dob"], data["insurance_id"]))
    patient_id = c.execute("SELECT id FROM patients WHERE insurance_id=?", (data["insurance_id"],)).fetchone()[0]
    c.execute("INSERT INTO documents (patient_id, visit_date, raw_text) VALUES (?, ?, ?)", 
              (patient_id, data["visit_date"], raw_text))
    doc_id = c.lastrowid
    entities = []
    for d in data.get("diagnoses", []): entities.append(("Diagnose", d))
    for s in data.get("symptoms", []): entities.append(("Symptom", s))
    for m in data.get("medications", []): entities.append(("Medikament", f"{m['name']} ({m['dosage']})"))
    for a in data.get("allergies", []): entities.append(("Allergie", a))
    for r in data.get("recommendations", []): entities.append(("Empfehlung", r))
    for etype, evalue in entities:
        c.execute("INSERT INTO medical_entities (document_id, entity_type, entity_value) VALUES (?, ?, ?)", (doc_id, etype, evalue))
    conn.commit()
    conn.close()

def process_letter(raw_text: str, tokenizer, model, device: str) -> str:
    init_db()
    try:
        print("🚀 Starting pipeline...")
        data = extract_json(raw_text, tokenizer, model, device)
        
        # Validate: check if model returned placeholders instead of real values
        placeholder_indicators = ["string", "DD.MM.YYYY", "[NAME", "[DATUM", "[VERSICHERUNG", "[BESUCHSDATUM"]
        for field in ["patient_name", "dob", "insurance_id"]:
            val = data.get(field, "")
            if val and any(ph in str(val) for ph in placeholder_indicators):
                return "❌ Model returned template placeholders instead of extracted values."
        
        # Also check if it's returning the same hardcoded example (Max Mustermann)
        if data.get("patient_name") == "Max Mustermann" and "A1234567890" in str(data.get("insurance_id", "")):
            return "❌ Model is returning example data instead of extracting from the letter. The prompt needs adjustment."
        
        required = ["patient_name", "insurance_id", "visit_date"]
        missing = [k for k in required if not data.get(k)]
        if missing:
            return f"❌ Missing required fields: {missing}"
        
        save_to_db(data, raw_text)
        print(f"✅ Pipeline completed. Patient: {data['patient_name']}")
        return f"✅ Success! Patient '{data['patient_name']}' processed and saved."
    except json.JSONDecodeError as e:
        return f"❌ JSON parse error: {str(e)}"
    except Exception as e:
        error_msg = f"❌ Pipeline error: {type(e).__name__}: {str(e)}"
        print(error_msg)
        return error_msg