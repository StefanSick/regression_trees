
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
import itertools
from reg_package.eval import evaluate_regression
from reg_package.rf import RegRF
from reg_package.tree import RegTree

def preprocess_train_val(X_train, X_val, num_cols, cat_cols):
    """
    - Numeric: KNN imputation
    - Categorical: Most frequent
    """

    X_train = X_train.copy()
    X_val   = X_val.copy()
    # categorical
    if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            X_val[cat_cols]   = cat_imputer.transform(X_val[cat_cols])

    # numeric
    if num_cols:
        num_imputer = KNNImputer(n_neighbors=5)
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_val[num_cols]   = num_imputer.transform(X_val[num_cols])
    return X_train, X_val

def cross_val_rmse(model_class, X, y, num_cols, cat_cols,
                  cv=5, random_state=42, **model_kwargs):

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_train_cv = X.iloc[train_idx]
        X_val_cv   = X.iloc[val_idx]
        y_train_cv = y.iloc[train_idx].values
        y_val_cv   = y.iloc[val_idx].values

        # Preprocessing
        X_train_cv, X_val_cv = preprocess_train_val(
            X_train_cv, X_val_cv, num_cols, cat_cols
        )

        # Convert after preprocessing
        X_train_arr = np.array(X_train_cv, dtype=object)
        X_val_arr   = np.array(X_val_cv, dtype=object)

        model = model_class(**model_kwargs)
        model.fit(X_train_arr, y_train_cv)
        preds = model.predict(X_val_arr)

        rmse = np.sqrt(np.mean((y_val_cv - preds) ** 2))
        rmses.append(rmse)

    return np.array(rmses)


def train_eval_rf(name, df, target_col, cat_cols):
    print(f"\n=== Processing Dataset: {name} ===")
    results_list = []
    # Setup
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # Get indices for categorical columns 
    cat_features_idx = [X.columns.get_loc(c) for c in cat_cols]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define Grid
    param_grid = {
        'max_depth': [5, 10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'max_features': [5, 10] 
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} combinations with 5-fold CV...")

    best_score = float("inf")
    best_params = None
    
    # Search Loop
    for params in combinations:
        
        print(f"Testing: {params} ...", end=" ")
        
        cv_rmses = cross_val_rmse(
            RegRF,
            X_train,
            y_train,
            num_cols=num_cols,
            cat_cols=cat_cols,
            cv=5,
            n_estimators=50, 
            cat_features=cat_features_idx,
            **params 
        )
        
        mean_rmse = cv_rmses.mean()

        # create a result list
        print(f"RMSE: {mean_rmse:.4f}")
        result_row = params.copy()
        result_row['rmse'] = mean_rmse
        results_list.append(result_row)
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    print(f"--> Winner: {best_params} with CV RMSE {best_score:.4f}")
    results_df = pd.DataFrame(results_list)

    # Sort best models are at the top
    results_df = results_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)
    # training
    X_train_proc, X_test_proc = preprocess_train_val(
        X_train, X_test, num_cols, cat_cols
    )
    
    X_train_arr = np.array(X_train_proc, dtype=object)
    X_test_arr  = np.array(X_test_proc, dtype=object)

    # Train final model with BEST params
    final_rf = RegRF(
        n_estimators=50,
        cat_features=cat_features_idx,
        random_state=42,
        **best_params 
    )
    
    final_rf.fit(X_train_arr, y_train.values)
    preds = final_rf.predict(X_test_arr)

    metrics = evaluate_regression(y_test.values, preds)
    print(f"Final Test Metrics: {metrics}")
    
    return final_rf, results_df


def train_eval_tree(name, df, target_col, cat_cols, param_grid=None):
    """
    Runs Grid Search CV and final evaluation for a single Regression Tree.
    """
    print(f"\n=== Processing Dataset: {name} ===")
    results_list = []
    
    # setup
        
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

        
    num_cols = [c for c in X.columns if c not in cat_cols]
    # Get indices for categorical columns 
    cat_features_idx = [X.columns.get_loc(c) for c in cat_cols]
    


    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5]
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Testing {len(combinations)} combinations with 5-fold CV...")

    best_score = float("inf")
    best_params = None

    # Grid search loop
    for params in combinations:
        
        cv_rmses = cross_val_rmse(
            RegTree,
            X_train,
            y_train,
            num_cols=num_cols,
            cat_cols=cat_cols,
            cv=5,
            cat_features=cat_features_idx,
            **params 
        )
        
        mean_rmse = cv_rmses.mean()
        
        # Save results
        result_row = params.copy()
        result_row['rmse'] = mean_rmse
        results_list.append(result_row)

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    print(f"--> Winner: {best_params}")
    print(f"--> Best CV RMSE: {best_score:.4f}")

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)

    # Training with best params

    X_train_proc, X_test_proc = preprocess_train_val(
        X_train, X_test, num_cols, cat_cols
    )

   
    if isinstance(X_train_proc, pd.DataFrame):
        current_cols = X_train_proc.columns
        cat_features_idx = [current_cols.get_loc(c) for c in cat_cols if c in current_cols]

    # Convert to numpy object arrays
    X_train_arr = np.array(X_train_proc, dtype=object)
    X_test_arr  = np.array(X_test_proc, dtype=object)

    print("Training Final Regression Tree...")
    final_tree = RegTree(
        cat_features=cat_features_idx,
        **best_params 
    )
    
    final_tree.fit(X_train_arr, y_train.values)
    preds = final_tree.predict(X_test_arr)

    metrics = evaluate_regression(y_test.values, preds)
    print("Final Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return final_tree, results_df


def drop_correlated(df, threshold):
    """
    Drop correlated columns
    """
    columns_to_drop = set()          
    corr = df.corr()
                      
    for i in range(len(corr.columns)):
        for j in range(i):            
            if abs(corr.iloc[i, j]) > threshold:  
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop