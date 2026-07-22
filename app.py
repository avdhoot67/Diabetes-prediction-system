from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load("diabetes_decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset to compute mean values for auto-fill and scaling
df = pd.read_csv("diabetes_new2.csv")
# Use all columns except Gender and CLASS (same as training)
numeric_cols = ['AGE','Urea','Cr','Glucose','Chol','TG','HDL','LDL','VLDL','BMI']

# Apply same preprocessing as training
cols_to_replace = ['Glucose','BMI','Urea','Cr','Chol','TG','HDL','LDL','VLDL']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
for col in cols_to_replace:
    df[col].fillna(df[col].mean(), inplace=True)

mean_values = df[numeric_cols].mean().to_dict()

# Scaler is already loaded above

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/test')
def test():
    return render_template('index.html', errors={}, values={})

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_values = {}
    errors = {}

    # Define valid ranges for each field
    ranges = {
        'AGE': {'min': 0, 'max': 150, 'unit': 'years'},
        'Urea': {'min': 2.0, 'max': 8.0, 'unit': 'mmol/L'},
        'Cr': {'min': 40, 'max': 120, 'unit': 'μmol/L'},
        'Glucose': {'min': 70, 'max': 200, 'unit': 'mg/dL'},
        'Chol': {'min': 3.0, 'max': 7.0, 'unit': 'mmol/L'},
        'TG': {'min': 0.5, 'max': 6.0, 'unit': 'mmol/L'},
        'HDL': {'min': 0.5, 'max': 4.0, 'unit': 'mmol/L'},
        'LDL': {'min': 0.3, 'max': 4.9, 'unit': 'mmol/L'},
        'VLDL': {'min': 0.2, 'max': 16.0, 'unit': 'mmol/L'},
        'BMI': {'min': 15, 'max': 50, 'unit': 'kg/m²'}
    }
    
    # Collect form data - no empty fields allowed
    for col in numeric_cols:
        val = request.form.get(col, '').strip()
        if val == '':
            errors[col] = "This field is required"
        else:
            try:
                num_val = float(val)
                # Check range validation
                if col in ranges:
                    range_info = ranges[col]
                    if num_val < range_info['min'] or num_val > range_info['max']:
                        errors[col] = f"Value must be between {range_info['min']} and {range_info['max']} {range_info['unit']}"
                    else:
                        form_values[col] = num_val
                else:
                    form_values[col] = num_val
            except ValueError:
                errors[col] = "Please enter a valid number (integer or decimal)"

    if errors:
        return render_template('index.html', errors=errors, values=form_values)

    try:
        # Create feature array in same column order
        features = [form_values[col] for col in numeric_cols]
        
        # Apply same scaling as training
        features_scaled = scaler.transform([features])
        
        # Debug: Print features and prediction
        print(f"Original features: {features}")
        print(f"Scaled features: {features_scaled}")
        print(f"Scaled features shape: {features_scaled.shape}")
        
        prediction = model.predict(features_scaled)
        print(f"Raw prediction: {prediction}")

        # Map model output to class name (model outputs 0, 1, 2)
        predicted_class_num = prediction[0]
        print(f"Predicted class number: {predicted_class_num}")
        
        # Map numeric prediction to string labels (N=0, P=1, Y=2)
        class_mapping = {
            0: 'Non Diabetic',
            1: 'Prediabetic', 
            2: 'Diabetic'
        }
        
        predicted_class = class_mapping.get(predicted_class_num, 'Unknown')
        print(f"Predicted class string: {predicted_class}")

        return render_template('result.html', prediction=predicted_class)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        errors['__all__'] = str(e)
        return render_template('index.html', errors=errors, values=form_values)

if __name__ == "__main__":
    app.run(debug=True)
