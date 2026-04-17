# Medical Dashboard - Flask Application

A beautiful, modern medical dashboard for processing German doctor's letters using AI.

## Features

- **Modern UI**: Clean, professional interface with sidebar navigation
- **Patient Search**: Real-time search through all patients
- **Letter Upload**: Drag-and-drop upload for medical letters
- **AI Processing**: Automatic extraction of patient data using ML model
- **Patient Details**: Comprehensive view of diagnoses, symptoms, medications, allergies, and recommendations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your database exists in `/db/med_data.db`

## Running the Application

Navigate to the flask_app directory and run:

```bash
cd flask_app
python app.py
```

Then open your browser to: **http://localhost:5000**

## Navigation

- **Search**: View and search all patients
- **Doctoral Letter**: Upload and process new medical letters

## File Structure

```
flask_app/
├── app.py                 # Main Flask application
├── templates/
│   ├── base.html         # Base template with sidebar
│   ├── search.html       # Patient search page
│   ├── patient.html      # Patient details page
│   └── letter.html       # Letter upload page
├── static/
│   ├── css/
│   │   └── style.css     # Modern styling
│   └── js/
│       └── main.js       # JavaScript functionality
└── uploads/              # Temporary upload folder
```

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with modern design system
- **Icons**: Font Awesome 6
- **Fonts**: Inter (Google Fonts)
- **Database**: SQLite
- **ML Model**: Qwen2.5-1.5B-Instruct
