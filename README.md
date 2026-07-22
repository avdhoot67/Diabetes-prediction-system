# Diabetes Prediction (Flask App)

A Flask web application that predicts diabetes status (Non Diabetic, Prediabetic, Diabetic) from clinical inputs using a trained Decision Tree model. The app validates inputs, scales features consistently with training, and presents clear results.

## Features
- Input form for clinical metrics: `AGE`, `Urea`, `Cr`, `Glucose`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI`
- Client- and server-side validation with clear messages
- Preprocessing identical to training (zeros treated as missing; mean imputation for select columns)
- Feature scaling via pre-fitted scaler (`scaler.pkl`)
- Prediction via `diabetes_decision_tree_model.pkl`
- Polished UI with handy preset autofill buttons for demo

## Tech Stack
- Python 3.9+
- Flask
- NumPy, Pandas, scikit-learn, joblib

## Project Structure
```
.
├─ app.py                        # Flask server (routes, validation, inference)
├─ diabetes_decision_tree_model.pkl
├─ scaler.pkl
├─ diabetes_new2.csv             # Dataset used to compute mean values
├─ templates/
│  ├─ index.html                 # Input form page
│  ├─ result.html                # Result page
│  ├─ landing.html               # Landing page
│  └─ info.html                  # Info page (optional)
└─ static/                       # Images, favicon
```

## Setup
### 1) Clone the repository
```bash
git clone <your-repo-url>
cd Diabetes_prediction_advanced
```

### 2) Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1

# macOS/Linux (bash)
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, create one with at least:
```txt
flask
joblib
numpy
pandas
scikit-learn
```

### 4) Ensure model artifacts exist
Make sure the following files are present in the project root:
- `diabetes_decision_tree_model.pkl`
- `scaler.pkl`
- `diabetes_new2.csv`

If missing, retrain/export them using your notebooks (e.g., `train.ipynb`) or request them from the author.

## Run Locally
```bash
# From project root, with venv activated
python app.py
```
The server starts at `http://127.0.0.1:5000/`.
- Home: `/` (landing page)
- Form: `/test`
- Info: `/info` (if present)

## How It Works
- `app.py` loads the trained model and scaler using `joblib`
- Computes column means from `diabetes_new2.csv` and applies training-like preprocessing (replace zeros with NaN, then mean-impute selected columns)
- Validates incoming form values for numeric format and physiological ranges
- Scales features using the pre-fitted scaler and predicts via the model
- Maps numeric prediction to `Non Diabetic` / `Prediabetic` / `Diabetic` and renders `result.html`

## Environment
No special environment variables required for local development. The app runs with Flask `debug=True` by default.

## Deploy (Optional)
You can deploy on Render, Railway, Azure App Service, or a VPS.
1. Set `debug=False` in `app.py` for production.
2. Optionally expose host/port from environment:
   ```python
   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
   ```
3. Ensure model artifacts (`.pkl`) and `diabetes_new2.csv` are deployed with the app.
4. Provide a proper `requirements.txt` and platform config (e.g., `Procfile` if required).

## Push This Project to GitHub
1. Initialize (or reuse) git in the project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Diabetes prediction app"
   ```
2. Create an empty repo on GitHub (avoid auto-adding README/License to prevent conflicts).
3. Add the remote and push:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M master
   git push -u origin master
   ```

## How Others Can Access/Run This Project
- Clone and run locally:
  ```bash
  git clone https://github.com/<your-username>/<your-repo>.git
  cd <your-repo>
  python -m venv .venv
  . .venv\Scripts\Activate.ps1   # Windows
  pip install -r requirements.txt
  python app.py
  ```
  Then open `http://127.0.0.1:5000/` in a browser.

- Or deploy to a hosting service and share the URL. Ensure `diabetes_decision_tree_model.pkl`, `scaler.pkl`, and `diabetes_new2.csv` are included in deployment assets.

## Troubleshooting
- Import errors: Install dependencies inside the active virtual environment.
- Missing files: Confirm `.pkl` and CSV exist at the project root with exact names used in `app.py`.
- Port in use: Change the port or stop the process using it.

## License
Add your preferred license (e.g., MIT) in a `LICENSE` file.

## Acknowledgements
- Built with Flask and scikit-learn.
