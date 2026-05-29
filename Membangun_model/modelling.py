import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv(
    "Membangun_model/data_konstruksi_preprocessed/preprocessed.csv"
)

X = df.drop("Status", axis=1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(
    classification_report(
        y_test,
        pred
    )
)

joblib.dump(
    model,
    "outputs/model_proyek.pkl"
)

print("Model baseline tersimpan")