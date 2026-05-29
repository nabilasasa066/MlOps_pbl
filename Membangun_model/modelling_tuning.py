from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

param_grid = {
    "n_estimators": [50,100,200],
    "max_depth": [5,10,None],
    "min_samples_split": [2,5]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

joblib.dump(
    best_model,
    "outputs/model_proyek_tuned.pkl"
)

print(grid.best_params_)