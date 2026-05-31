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

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    mlflow.set_experiment("construction-project-classification")


    MODEL_DIR = "outputs"
    os.makedirs(MODEL_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_DIR, "model_proyek.pkl")


    df = pd.read_csv(args.data_path)

    if "Status" not in df.columns:
        raise ValueError("Kolom 'Status' tidak ditemukan di dataset!")

    X = df.drop(columns=["Status"])
    y = df["Status"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    with mlflow.start_run(run_name="random_forest_konstruksi"):

  
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)


        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(model, MODEL_PATH)

        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.log_artifact(MODEL_PATH)

        print("=================================")
        print("MODEL TRAINING DONE SUCCESSFULLY")
        print("Model saved at:", MODEL_PATH)
        print("=================================")