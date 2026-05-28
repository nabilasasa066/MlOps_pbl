import argparse
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)

# --- MLflow setup via environment variables ---
# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
# diset otomatis oleh GitHub Actions secrets — tidak perlu dagshub.init()
tracking_uri = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/RFer7935/MLOps-Experiment.mlflow"
)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("sales-value-classification")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../preprocessing/data_penjualan_preprocessed/preprocessed.csv"
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    with mlflow.start_run():
        # --- Log parameters ---
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("data_path", args.data_path)

        # --- Load data ---
        df = pd.read_csv(args.data_path)
        X = df.drop(columns=["High_Value"])
        y = df["High_Value"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Train model ---
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(classification_report(y_test, y_pred, target_names=["Low Value", "High Value"]))
        print(f"AUC-ROC: {auc:.4f}")

        # --- Log metrics ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", auc)

        # --- Log model ---
        mlflow.sklearn.log_model(model, "model")

        # --- Save model artifact locally ---
        os.makedirs("outputs", exist_ok=True)
        joblib.dump(model, "outputs/rf_model.pkl")
        mlflow.log_artifact("outputs/rf_model.pkl")

        print(f"[INFO] Run logged to DagsHub MLflow.")
        print(f"[INFO] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
