import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_or_generate_data():
    """Loads existing dataset or generates realistic dataset if not present."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # If dataset is small, expand to 1500 records for optimal model evaluation
        if len(df) < 500:
            print("Expanding dataset to 1,500 records for robust training...")
            from data.generate_data import generate_credit_dataset
            df = generate_credit_dataset(n_samples=1500)
            df.to_csv(DATA_PATH, index=False)
    else:
        from data.generate_data import generate_credit_dataset
        df = generate_credit_dataset(n_samples=1500)
        df.to_csv(DATA_PATH, index=False)
        
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution (Creditworthy): \n{df['Creditworthy'].value_counts(normalize=True).round(3)}")
    return df

def preprocess_data(df):
    """Encodes categorical features and scales numerical features."""
    X = df.drop(columns=["Creditworthy"])
    y = df["Creditworthy"]

    # Identify numeric and categorical columns
    categorical_cols = ["Loan_Purpose"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = encoder.fit_transform(X[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)

    # Scale numeric features
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(X[numeric_cols])

    # Combine processed features
    X_processed = np.hstack([num_scaled, cat_encoded])
    all_feature_names = list(numeric_cols) + list(cat_feature_names)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.25, random_state=42, stratify=y
    )

    metadata = {
        "scaler": scaler,
        "encoder": encoder,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": all_feature_names,
    }

    return X_train, X_test, y_train, y_test, metadata

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Trains multiple classifiers and compares their performances."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    roc_data = {}
    best_model_name = None
    best_f1 = -1

    print("\n" + "=" * 65)
    print("           MODEL TRAINING AND BENCHMARK RESULTS")
    print("=" * 65)

    plt.figure(figsize=(9, 6))

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "model": model,
            "y_pred": y_pred
        }

        print(f"\nModel: {name}")
        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # Complete ROC Curve Plot
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    plt.title("ROC-AUC Comparison Curve - Credit Scoring Models", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    roc_path = os.path.join(PLOT_DIR, "roc_curve_comparison.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"\n[Plot Saved] ROC-AUC curve saved to: {roc_path}")

    # Best Model Evaluation details
    best_info = results[best_model_name]
    print(f"\n>>> Best Performing Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, best_info["y_pred"], target_names=["Denied (0)", "Approved (1)"]))

    # Confusion Matrix for Best Model
    cm = confusion_matrix(y_test, best_info["y_pred"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Denied (0)", "Approved (1)"],
                yticklabels=["Denied (0)", "Approved (1)"])
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    cm_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Confusion matrix saved to: {cm_path}")

    # Feature Importance (if applicable)
    best_model = best_info["model"]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        plt.figure(figsize=(9, 5))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
        plt.title(f"Top 10 Feature Importances ({best_model_name})", fontsize=13, fontweight="bold")
        plt.xlabel("Relative Importance")
        fi_path = os.path.join(PLOT_DIR, "feature_importance.png")
        plt.tight_layout()
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[Plot Saved] Feature importance plot saved to: {fi_path}")

    return best_model_name, best_info["model"]

def main():
    print("Starting Credit Scoring Model Training Pipeline...")
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test, metadata = preprocess_data(df)

    best_name, best_model = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, metadata["feature_names"]
    )

    # Save artifacts for inference and web app
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": metadata["scaler"],
        "encoder": metadata["encoder"],
        "numeric_cols": metadata["numeric_cols"],
        "categorical_cols": metadata["categorical_cols"],
    }
    model_save_path = os.path.join(MODEL_DIR, "credit_model.pkl")
    joblib.dump(artifact, model_save_path)
    print(f"\n[Model Saved] Best model pipeline successfully saved to: {model_save_path}")

if __name__ == "__main__":
    main()
