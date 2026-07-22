import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_artifacts():
    model = joblib.load("diabetes_decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def load_and_preprocess(scaler):
    df = pd.read_csv("diabetes_new2.csv")
    feature_columns = ['AGE','Urea','Cr','Glucose','Chol','TG','HDL','LDL','VLDL','BMI']

    # Mirror preprocessing from app.py
    cols_to_replace = ['Glucose','BMI','Urea','Cr','Chol','TG','HDL','LDL','VLDL']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
    for col in cols_to_replace:
        df[col].fillna(df[col].mean(), inplace=True)

    X = df[feature_columns].copy()

    # Labels mapping in dataset: CLASS likely values 'N', 'P', 'Y'
    # Map to numeric classes used in app.py (0,1,2)
    label_map = {'N': 0, 'P': 1, 'Y': 2}
    y = df['CLASS'].map(label_map)

    # Filter rows with valid labels
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    X_scaled = scaler.transform(X.values)
    return X_scaled, y


def main():
    # Try loading saved artifacts; if that fails (e.g., numpy ABI mismatch), train fresh
    try:
        model, scaler = load_artifacts()
        X_scaled, y_true = load_and_preprocess(scaler)
        y_pred = model.predict(X_scaled)
    except Exception as e:
        print(f"Warning: failed to load saved artifacts: {e}")
        print("Falling back to training a fresh model for evaluation...")

        df = pd.read_csv("diabetes_new2.csv")
        feature_columns = ['AGE','Urea','Cr','Glucose','Chol','TG','HDL','LDL','VLDL','BMI']
        cols_to_replace = ['Glucose','BMI','Urea','Cr','Chol','TG','HDL','LDL','VLDL']
        df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
        for col in cols_to_replace:
            df[col].fillna(df[col].mean(), inplace=True)

        X = df[feature_columns].copy()
        y = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
        mask = y.notna()
        X = X[mask]
        y = y[mask].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_true = y_test
        y_pred = model.predict(X_test_scaled)

    # Prepare report
    target_names = ['Non Diabetic', 'Prediabetic', 'Diabetic']
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    correct = int((y_true == y_pred).sum())
    total = int(len(y_true))

    # Print table-like output
    print("Performance Report")
    print("Class\tPrecision\tRecall\tF1-Score\tSupport")
    for i, name in enumerate(target_names):
        cls = str(i)
        row = report_dict.get(name)
        if row is None:
            # fallback when target_names are not keys in output_dict
            row = report_dict.get(cls, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        print(f"{name}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1-score']:.4f}\t{int(row['support'])}")

    print("\nAverages")
    for avg_key in ["micro avg", "macro avg", "weighted avg"]:
        row = report_dict.get(avg_key) or report_dict.get(avg_key.replace(" ", "_"), None)
        if row:
            print(f"{avg_key}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1-score']:.4f}\t{int(row['support'])}")

    print(f"\nAccuracy\t{accuracy:.4f}")
    print(f"Correct predictions\t{correct}/{total}")


if __name__ == "__main__":
    main()


