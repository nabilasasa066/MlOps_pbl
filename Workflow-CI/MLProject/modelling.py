import argparse
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)

    args = parser.parse_args()

    print("Loading data...")

    df = pd.read_csv(args.data_path)

    X = df.drop(columns=["Status"])
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model_proyek.pkl")

    print("MODEL SAVED SUCCESS")