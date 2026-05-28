import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# --- Load dataset ---
df = pd.read_csv("preprocessing/data_penjualan_preprocessed/preprocessed.csv")

X = df.drop(columns=["High_Value"])
y = df["High_Value"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Hyperparameter tuning ---
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best F1 (CV):", grid_search.best_score_)

# --- Evaluate best model ---
y_pred = grid_search.best_estimator_.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Low Value", "High Value"])
print("Classification Report:\n", report)

# --- Save best model ---
joblib.dump(grid_search.best_estimator_, "Membangun_model/rf_model_tuned.pkl")
print("[INFO] Tuned model saved to Membangun_model/rf_model_tuned.pkl")
