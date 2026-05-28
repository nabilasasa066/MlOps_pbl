import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Load dataset ---
df = pd.read_csv("preprocessing/data_penjualan_preprocessed/preprocessed.csv")

X = df.drop(columns=["High_Value"])
y = df["High_Value"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Build baseline model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Low Value", "High Value"])
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model ---
joblib.dump(model, "Membangun_model/rf_model.pkl")
print("[INFO] Model saved to Membangun_model/rf_model.pkl")
