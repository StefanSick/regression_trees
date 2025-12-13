import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PATH_DATA = ROOT / "data"

data_frames ={}
for file_path in PATH_DATA.glob("*.csv"):
    key_name = file_path.stem 
    df = pd.read_csv(file_path)
    data_frames[key_name] = df
    print(f"Geladen: {key_name} -> Shape: {df.shape}")


# df_cancer = data_frames["cancer_reg"].drop(columns= ["pctsomecol18_24"]) # had to drop pctsomecol18_24 because of the amount of missing data
# df_cancer.isnull().sum()
# impute_cols = ['pctemployed16_over', 'pctprivatecoveragealone']
# imputer = KNNImputer(n_neighbors=5)
# df_cancer[impute_cols] = imputer.fit_transform(df_cancer[impute_cols])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Index of split feature
        self.threshold = threshold  # Split value
        self.left = left            # Left side
        self.right = right          # Right side
        self.value = value          # Mean of target value

    def is_leaf(self):
        return self.value is not None

class RegTree:
    def __init__(self, max_depth=10, min_samples_split=2, cat_features=None):

        self.max_depth = max_depth # Max depth of tree
        self.min_samples_split = min_samples_split # Threshold for split
        self.cat_features = set(cat_features) if cat_features else set() # List of all categorical features
        self.root = None

    def fit(self, X, y):
        # Use object so int and float works
        X = np.array(X, dtype=object) 
        y = np.array(y, dtype=float)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Stopping Criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) < 1e-8):
            return Node(value=np.mean(y))

        # Best Split
        feat_idx, thresh = self._best_split(X, y, n_features)

        # If no valid split reduces error, return leaf
        if feat_idx is None:
            return Node(value=np.mean(y))

        # Create Children
        left_idxs, right_idxs = self._split(X[:, feat_idx], thresh, feat_idx)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=feat_idx, threshold=thresh, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_reduction = -1
        split_idx, split_thresh = None, None
        
        # Calculate current MSE
        parent_mse = self._mean_squared_error(y)

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            
            if feat_idx not in self.cat_features:
                X_column = X_column.astype(float)

            unique_values = np.unique(X_column)
            
            if len(unique_values) > 100:
                unique_values = np.percentile(unique_values, [25, 50, 75])

            for threshold in unique_values:
                # Calculate MSE Reduction
                reduction = self._mse_reduction(y, X_column, threshold, parent_mse, feat_idx)

                if reduction > best_reduction:
                    best_reduction = reduction
                    split_idx = feat_idx
                    split_thresh = threshold
                    
        return split_idx, split_thresh

    def _mse_reduction(self, y, X_column, threshold, parent_mse, feat_idx):
        # Split
        left_idxs, right_idxs = self._split(X_column, threshold, feat_idx)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Weighted Average of Children MSE
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        mse_l = self._mean_squared_error(y[left_idxs])
        mse_r = self._mean_squared_error(y[right_idxs])
        
        child_mse = (n_l / n) * mse_l + (n_r / n) * mse_r

        return parent_mse - child_mse

    def _mean_squared_error(self, y):
        if len(y) == 0: return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _split(self, X_column, threshold, feat_idx):
        if feat_idx in self.cat_features:
            # Equality Split for Categorical
            left_idxs = np.where(X_column == threshold)[0]
            right_idxs = np.where(X_column != threshold)[0]
        else:
            # Magnitude Split for Numerical
            X_column = X_column.astype(float) 
            left_idxs = np.where(X_column <= threshold)[0]
            right_idxs = np.where(X_column > threshold)[0]
        
        return left_idxs, right_idxs

    def predict(self, X):
        X = np.array(X, dtype=object)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if node.feature in self.cat_features:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            # Float comparison for numerical features
            val = float(x[node.feature])
            if val <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

def preprocess_train_val(X_train, X_val, num_cols, cat_cols):
    """
    - Numeric: KNN imputation
    - Categorical: Most frequent
    """

    X_train = X_train.copy()
    X_val   = X_val.copy()

    # categorical
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_val[cat_cols]   = cat_imputer.transform(X_val[cat_cols])

    # numeric
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

        # preprocessing
        X_train_cv, X_val_cv = preprocess_train_val(
            X_train_cv, X_val_cv, num_cols, cat_cols
        )

        # convert after preprocessing
        X_train_arr = np.array(X_train_cv, dtype=object)
        X_val_arr   = np.array(X_val_cv, dtype=object)

        model = model_class(**model_kwargs)
        model.fit(X_train_arr, y_train_cv)
        preds = model.predict(X_val_arr)

        rmse = np.sqrt(np.mean((y_val_cv - preds) ** 2))
        rmses.append(rmse)

    return np.array(rmses)

def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression predictions with multiple metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }

if __name__ == "__main__":
    df_possum = data_frames["possum"].drop(columns=["case"])

    # drop missing target columns
    df_possum = df_possum.dropna(subset=["footlgth"])

    y = df_possum["footlgth"]
    X = df_possum.drop(columns=["footlgth"])

    # define categorical and numeric columns
    cat_cols = ["site", "sex", "Pop"]    
    num_cols = list(X.columns.difference(cat_cols))
    
    cat_features = [X.columns.get_loc(c) for c in cat_cols]
   
    cv_rmses = cross_val_rmse(
        RegTree,
        X,
        y,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cv=5,
        random_state=42,
        max_depth=5,
        min_samples_split=2,
        cat_features=cat_features,
    )
    print("CV RMSEs:", cv_rmses)
    print("Mean CV RMSE:", cv_rmses.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # preprocessing 
    X_train, X_test = preprocess_train_val(
        X_train, X_test, num_cols, cat_cols
    )

    # convert to numpy object arrays
    X_train_arr = np.array(X_train, dtype=object)
    X_test_arr  = np.array(X_test, dtype=object)

    reg_tree = RegTree(max_depth=5, cat_features=cat_features)
    
    print("Training Regression Tree...")
    reg_tree.fit(X_train, y_train)
    
    # Predict the values
    preds = reg_tree.predict(X_test_arr)

    metrics = evaluate_regression(y_test.values, preds)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")