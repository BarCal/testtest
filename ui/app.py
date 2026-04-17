# ui/app.py
import streamlit as st
import sqlite3, pandas as pd, sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import load_model, process_letter

st.set_page_config(page_title="🩺 German Medical Dashboard", layout="wide")
st.title("🩺 Arztbrief Dashboard (MVP)")

@st.cache_resource
def get_model():
    return load_model()

tokenizer, model, device = get_model()
st.sidebar.info(f"🔌 Running on: {device.upper()}")

st.sidebar.header("📤 Dokument hochladen")
uploaded_file = st.sidebar.file_uploader("Wählen Sie eine .txt-Datei:", type=["txt"])

if uploaded_file is not None:
    with st.spinner("🔄 Verarbeite Dokument..."):
        raw_text = uploaded_file.read().decode("utf-8")
        result = process_letter(raw_text, tokenizer, model, device)
        st.sidebar.success(result)
        st.rerun()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "med_data.db"
if not DB_PATH.exists():
    st.info("📦 Datenbank wird beim ersten Upload automatisch erstellt.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
patients = pd.read_sql("SELECT * FROM patients", conn)
documents = pd.read_sql("SELECT * FROM documents", conn)
entities = pd.read_sql("SELECT * FROM medical_entities", conn)
conn.close()

if patients.empty:
    st.info("⚠️ Noch keine Patienten in der Datenbank. Bitte laden Sie einen Arztbrief hoch.")
    st.stop()

selected_name = st.sidebar.selectbox("Patient auswählen:", patients["name"].tolist())
selected_patient = patients[patients["name"] == selected_name].iloc[0]

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("👤 Persönliche Daten")
    st.markdown(f"**Name:** {selected_patient['name']}")
    st.markdown(f"**Geburtsdatum:** {selected_patient['dob']}")
    st.markdown(f"**Versichertennummer:** {selected_patient['insurance_id']}")
with col2:
    st.subheader("📄 Dokument")
    doc = documents[documents["patient_id"] == selected_patient["id"]].iloc[0]
    st.markdown(f"**Besuchsdatum:** {doc['visit_date']}")
    with st.expander("📜 Originaltext anzeigen"):
        st.text(doc["raw_text"])

st.divider()
st.subheader("🏥 Medizinische Entitäten")
patient_entities = entities[entities["document_id"] == doc["id"]]
if not patient_entities.empty:
    for etype, group in patient_entities.groupby("entity_type"):
        st.markdown(f"**{etype}:**")
        for _, row in group.iterrows():
            st.markdown(f"• {row['entity_value']}")
else:
    st.info("Keine medizinischen Entitäten gefunden.")