# COVID-19 X-Ray Analyzer - Setup Guide

## Quick Setup for Your Friend

### 1. Extract the ZIP file
Extract all files to a folder (e.g., `covid19-analyzer`)

### 2. Install Python
Make sure Python 3.8+ is installed on your computer
- Download from: https://python.org/downloads/
- During installation, check "Add Python to PATH"

### 3. Open Command Prompt/Terminal
Navigate to the project folder:
```bash
cd path/to/covid19-analyzer
```

### 4. Create Virtual Environment
```bash
python -m venv .venv
```

### 5. Activate Virtual Environment
**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 6. Install Dependencies
```bash
pip install -r requirements.txt
```

### 7. Run the Application
```bash
python app.py
```

### 8. Open Web Browser
Go to: `http://127.0.0.1:5000`

## What's Included
- `app.py` - Main Flask application
- `best_model.pth` - Trained AI model (45MB+)
- `templates/index.html` - Web interface
- `requirements.txt` - Python dependencies
- `README.md` - Detailed documentation

## System Requirements
- Python 3.8 or higher
- 2GB+ RAM
- Internet connection (for initial setup)

## Troubleshooting
- If PyTorch installation fails, try: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- If port 5000 is busy, the app will show an alternative port
- Make sure all files are in the same folder

## Features
✅ AI-powered COVID-19 detection from chest X-rays  
✅ Professional medical interface  
✅ Drag & drop file upload  
✅ Real-time analysis with confidence scores  
✅ Detailed probability breakdown  

## Medical Disclaimer
This tool is for educational purposes only. Not for actual medical diagnosis.

---
Created by: Your Name
Date: August 2025
