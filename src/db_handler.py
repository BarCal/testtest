import sqlite3
import json
from pathlib import Path

# Robust path handling (works on Windows & Linux/Mac)
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "med_data.db"
DB_PATH.parent.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, dob TEXT, insurance_id TEXT UNIQUE
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER, visit_date TEXT, raw_text TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS medical_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER, entity_type TEXT, entity_value TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )''')
    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DB_PATH)

def insert_data(json_file: str, txt_file: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(txt_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 1. Insert Patient
    c.execute("INSERT OR IGNORE INTO patients (name, dob, insurance_id) VALUES (?, ?, ?)",
              (data["patient_name"], data["dob"], data["insurance_id"]))
    patient_id = c.execute("SELECT id FROM patients WHERE insurance_id=?", (data["insurance_id"],)).fetchone()[0]

    # 2. Insert Document
    c.execute("INSERT INTO documents (patient_id, visit_date, raw_text) VALUES (?, ?, ?)",
              (patient_id, data["visit_date"], raw_text))
    doc_id = c.lastrowid

    # 3. Flatten & Insert Medical Entities
    entities = []
    for d in data.get("diagnoses", []): entities.append(("Diagnose", d))
    for s in data.get("symptoms", []): entities.append(("Symptom", s))
    for m in data.get("medications", []): 
        entities.append(("Medikament", f"{m['name']} ({m['dosage']})"))
    for a in data.get("allergies", []): entities.append(("Allergie", a))
    for r in data.get("recommendations", []): entities.append(("Empfehlung", r))

    for etype, evalue in entities:
        c.execute("INSERT INTO medical_entities (document_id, entity_type, entity_value) VALUES (?, ?, ?)",
                  (doc_id, etype, evalue))

    conn.commit()
    conn.close()
    print(f"✅ Data inserted. Patient ID: {patient_id} | Document ID: {doc_id}")

if __name__ == "__main__":
    init_db()
    insert_data(
        BASE_DIR / "data" / "extracted.json",
        BASE_DIR / "data" / "letter_01.txt"
    )