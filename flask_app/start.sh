#!/bin/bash
# Start the Medical Dashboard Flask Application

echo "🚀 Starting Medical Dashboard..."
echo ""

cd "$(dirname "$0")"

# Check if database exists, if not initialize it
if [ ! -f "../db/med_data.db" ]; then
    echo "📦 Initializing database..."
    python -c "from src.pipeline import init_db; init_db()"
fi

# Start Flask app
echo "🌐 Launching application at http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""
python app.py
