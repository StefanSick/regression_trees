import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score, mean_absolute_error)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

warnings.simplefilter(action='ignore', category=FutureWarning)
  
ROOT = Path(__file__).resolve().parents[1]
PATH_DATA = ROOT / "data"

data_frames ={}
for file_path in PATH_DATA.glob("*.csv"):
    key_name = file_path.stem 
    df = pd.read_csv(file_path)
    data_frames[key_name] = df
    print(f"Geladen: {key_name} -> Shape: {df.shape}")

df_possum = data_frames["possum"].drop(columns=["case"])
df = df_possum

target_column = "footlgth"
X = df.drop(columns=[target_column])
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
y = df[target_column]

mask = ~y.isna()
X = X[mask]
y = y[mask]

test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

models = [
    (
        "DecisionTree",
        DecisionTreeRegressor(random_state=random_state),
        {
            "reg__max_depth": [None, 5, 10, 20],
            "reg__min_samples_split": [2, 5, 10],
            "reg__min_samples_leaf": [1, 2, 4],
        },
    ),
    (
        "SVR",
        SVR(),
        {
            "reg__kernel": ["rbf", "linear"],
            "reg__C": [0.1, 1, 10],
            "reg__epsilon": [0.01, 0.1, 1.0],
        },
    ),
    (
        "RandomForest",
        RandomForestRegressor(random_state=random_state),
        {
            "reg__n_estimators": [50, 100, 200],
            "reg__max_depth": [1, 5, 10],
            "reg__criterion": ["squared_error"],
            "reg__max_features": ["sqrt"],
        }
    ),
]

metrics = {
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}

def regression_fit(X_train, y_train, X_test, y_test, models, cv):
    rows = []
    total_steps = len(models)
    step = 1

    for name, reg, grid in models:
        print(f"\n=== Step {step}/{total_steps} | Model: {name} | ===")

        # Preprocessor -> Regressor
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("reg", reg),
            ]
        )

        grid_search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring=metrics,
            cv=cv,
            n_jobs=-1,
            refit="rmse",
            return_train_score=True,
            error_score="raise"
        )

        t0 = time.time()
        grid_search.fit(X_train, y_train)
        fit_time = time.time() - t0

        best = grid_search.best_estimator_
        best_idx = grid_search.best_index_

        row = {
            "model": name,
            "best_params": grid_search.best_params_,
            "fit_time": round(fit_time, 3),
        }

        # mean CV metrics at best params
        for m in metrics.keys():
            row[f"cv_{m}_mean"] = grid_search.cv_results_[f"mean_test_{m}"][best_idx]
            row[f"cv_{m}_std"] = grid_search.cv_results_[f"std_test_{m}"][best_idx]

        # test metrics
        yhat_test = best.predict(X_test)
        row.update(
            {
                "test_RMSE": root_mean_squared_error(y_test, yhat_test),
                "test_R2": r2_score(y_test, yhat_test),
                "test_MAE":  mean_absolute_error(y_test, yhat_test),
            }
        )

        rows.append(row)
        print(f"Done ({name})\n{'-'*60}")
        step += 1

    return pd.DataFrame(rows)

if __name__ == "__main__":
    results = regression_fit(
        X_train, y_train, X_test, y_test, models=models, cv=10
    )
    print("\nResults:")
    print(results)