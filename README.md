# DermAI Website

## Run locally

1. Open a terminal inside this folder.
2. Create a virtual environment (optional but recommended):
   python -m venv .venv
3. Activate it:
   Windows: .venv\Scripts\activate
   Mac/Linux: source .venv/bin/activate
4. Install dependencies:
   pip install -r requirements.txt
5. Run:
   python app.py
6. Open:
   http://127.0.0.1:5000

## Model details inspected
- Model file: best_skin_lesion_model.keras
- Input shape: (None, 224, 224, 3)
- Output: 1 sigmoid unit (binary classification)

IMPORTANT:
The website currently assumes output 1 corresponds to Malignant and output 0 corresponds to Benign.
Confirm this against the label encoding used in the training notebook before presenting results as clinically meaningful.

This project is an educational/research demonstration only and must not be used as a medical diagnostic tool.
