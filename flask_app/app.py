# Flask Medical Dashboard Application
import os
import sys
import json
import sqlite3
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import load_model, process_letter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-dashboard-secret-key-2024'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Global model variables
tokenizer = None
model = None
device = None

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    """Get database connection"""
    db_path = Path(__file__).parent.parent / "db" / "med_data.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_model():
    """Initialize the ML model globally"""
    global tokenizer, model, device
    if model is None:
        tokenizer, model, device = load_model()

@app.route('/')
def index():
    """Redirect to search page"""
    return redirect(url_for('search'))

@app.route('/search')
def search():
    """Search page - display all patients"""
    conn = get_db_connection()
    patients = conn.execute('SELECT * FROM patients ORDER BY name').fetchall()
    conn.close()
    return render_template('search.html', patients=patients)

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    """Patient information page"""
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    
    if patient is None:
        conn.close()
        return "Patient not found", 404
    
    documents = conn.execute('''
        SELECT * FROM documents WHERE patient_id = ? ORDER BY visit_date DESC
    ''', (patient_id,)).fetchall()
    
    # Get all medical entities for this patient's documents
    doc_ids = [doc['id'] for doc in documents]
    if doc_ids:
        placeholders = ','.join('?' * len(doc_ids))
        entities = conn.execute(f'''
            SELECT document_id, entity_type, entity_value FROM medical_entities 
            WHERE document_id IN ({placeholders})
            ORDER BY entity_type
        ''', doc_ids).fetchall()
    else:
        entities = []
    
    conn.close()
    return render_template('patient.html', patient=patient, documents=documents, entities=entities)

@app.route('/letter', methods=['GET', 'POST'])
def letter():
    """Doctoral letter upload and processing page"""
    error = None
    success = None
    result = None
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            error = "No file selected"
        else:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected"
            elif file and allowed_file(file.filename):
                # Initialize model if not already done
                try:
                    init_model()
                except Exception as e:
                    error = f"Failed to load model: {str(e)}"
                
                if not error:
                    # Save file temporarily
                    filename = secure_filename(file.filename)
                    filepath = app.config['UPLOAD_FOLDER'] / filename
                    file.save(filepath)
                    
                    # Read file content
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                        
                        # Process the letter
                        result = process_letter(raw_text, tokenizer, model, device)
                        
                        if result.startswith("✅"):
                            success = result
                            result = None
                        else:
                            error = result
                            result = None
                        
                        # Clean up uploaded file
                        filepath.unlink()
                        
                    except Exception as e:
                        error = f"Error processing file: {str(e)}"
                        if filepath.exists():
                            filepath.unlink()
            else:
                error = "File type not allowed. Please upload a .txt file."
    
    return render_template('letter.html', error=error, success=success, result=result)

@app.route('/api/search')
def api_search():
    """API endpoint for searching patients"""
    query = request.args.get('q', '')
    conn = get_db_connection()
    
    if query:
        patients = conn.execute('''
            SELECT * FROM patients 
            WHERE name LIKE ? OR insurance_id LIKE ? OR dob LIKE ?
            ORDER BY name
        ''', (f'%{query}%', f'%{query}%', f'%{query}%')).fetchall()
    else:
        patients = conn.execute('SELECT * FROM patients ORDER BY name').fetchall()
    
    conn.close()
    
    return jsonify([{
        'id': p['id'],
        'name': p['name'],
        'dob': p['dob'],
        'insurance_id': p['insurance_id']
    } for p in patients])

@app.route('/api/patient/<int:patient_id>')
def api_patient(patient_id):
    """API endpoint for patient details"""
    conn = get_db_connection()
    patient = conn.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    
    if patient is None:
        conn.close()
        return jsonify({'error': 'Patient not found'}), 404
    
    documents = conn.execute('''
        SELECT * FROM documents WHERE patient_id = ? ORDER BY visit_date DESC
    ''', (patient_id,)).fetchall()
    
    doc_ids = [doc['id'] for doc in documents]
    if doc_ids:
        placeholders = ','.join('?' * len(doc_ids))
        entities = conn.execute(f'''
            SELECT document_id, entity_type, entity_value FROM medical_entities 
            WHERE document_id IN ({placeholders})
        ''', doc_ids).fetchall()
    else:
        entities = []
    
    conn.close()
    
    return jsonify({
        'patient': dict(patient),
        'documents': [dict(doc) for doc in documents],
        'entities': [dict(ent) for ent in entities]
    })

if __name__ == '__main__':
    # Initialize database
    from src.pipeline import init_db as init_pipeline_db
    init_pipeline_db()
    
    print("🚀 Starting Flask Medical Dashboard...")
    print("📊 Access the dashboard at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
