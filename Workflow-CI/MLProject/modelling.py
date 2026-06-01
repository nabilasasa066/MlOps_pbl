import argparse
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

if __name__ == "__main__":

    # =========================
    # ARGUMENT PARSER
    # =========================
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="Membangun_model/data_konstruksi_preprocessed/preprocessed.csv"
    )

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)

    args = parser.parse_args()

    # =========================
    # DagsHub + MLflow Setup
    # =========================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI tidak ditemukan!")

    if not username:
        raise ValueError("DAGSHUB_USERNAME tidak ditemukan!")

    if not token:
        raise ValueError("DAGSHUB_TOKEN tidak ditemukan!")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(tracking_uri)

    print("=================================")
    print("MLFLOW TRACKING URI :", tracking_uri)
    print("DAGSHUB USERNAME    :", username)
    print("=================================")

    # =========================
    # EXPERIMENT
    # =========================
    mlflow.set_experiment(
        "construction-project-classification"
    )

    # =========================
    # OUTPUT DIRECTORY
    # =========================
    MODEL_DIR = "outputs"
    os.makedirs(MODEL_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "model_proyek.pkl"
    )

    # =========================
    # LOAD DATA
    # =========================
    print(f"Loading dataset: {args.data_path}")

    df = pd.read_csv(args.data_path)

    print("Dataset shape:", df.shape)

    if "Status" not in df.columns:
        raise ValueError(
            "Kolom target 'Status' tidak ditemukan!"
        )

    X = df.drop(columns=["Status"])
    y = df["Status"]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # START MLFLOW RUN
    # =========================
    with mlflow.start_run(
        run_name="random_forest_konstruksi"
    ):

        # Parameters
        mlflow.log_param(
            "n_estimators",
            args.n_estimators
        )

        mlflow.log_param(
            "max_depth",
            args.max_depth
        )

        mlflow.log_param(
            "min_samples_split",
            args.min_samples_split
        )

        # Model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        print(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model
        joblib.dump(model, MODEL_PATH)

        # Log model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Log artifact
        mlflow.log_artifact(MODEL_PATH)

        print("=================================")
        print("MODEL TRAINING SUCCESS")
        print("Accuracy :", acc)
        print("Model saved :", MODEL_PATH)
        print("=================================")