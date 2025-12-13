import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PATH_DATA = ROOT / "data"

data_frames ={}
for file_path in PATH_DATA.glob("*.csv"):
    key_name = file_path.stem 
    df = pd.read_csv(file_path)
    data_frames[key_name] = df
    print(f"Geladen: {key_name} -> Shape: {df.shape}")


df_cancer = data_frames["cancer_reg"].drop(columns= ["pctsomecol18_24"]) # had to drop pctsomecol18_24 because of the amount of missing data
df_cancer.isnull().sum()
impute_cols = ['pctemployed16_over', 'pctprivatecoveragealone']
imputer = KNNImputer(n_neighbors=5)
df_cancer[impute_cols] = imputer.fit_transform(df_cancer[impute_cols])


df_possum = data_frames["possum"].drop(columns=["case"])
impute_cols = ['age', 'footlgth']
imputer = KNNImputer(n_neighbors=5)
df_possum[impute_cols] = imputer.fit_transform(df_possum[impute_cols])


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

        # If no samples, just return a leaf with value 0 (or some default)
        if n_samples == 0:
            return Node(value=0.0)

        # Stopping Criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.std(y) == 0):
            return Node(value=float(np.mean(y)))

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
            

class RegRF:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 cat_features=None, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.cat_features = set(cat_features) if cat_features else set()
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []
        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.randint(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def _sample_features(self, n_features):
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return np.random.choice(n_features, self.max_features, replace=False)

    def fit(self, X, y):
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=float)

        n_features = X.shape[1]
        self.trees = []
        self.feature_subsets = []

        for _ in range(self.n_estimators):
            # bootstrap rows
            X_sample, y_sample = self._bootstrap_sample(X, y)
            # random subset of columns (features)
            feat_idx_subset = self._sample_features(n_features)
            self.feature_subsets.append(feat_idx_subset)

            # map categorical feature indices to the subset
            cat_in_subset = [np.where(feat_idx_subset == f)[0][0]
                             for f in self.cat_features if f in feat_idx_subset]

            tree = RegTree(max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split,
                           cat_features=cat_in_subset)
            tree.fit(X_sample[:, feat_idx_subset], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X, dtype=object)
        all_preds = []
        for tree, feat_idx_subset in zip(self.trees, self.feature_subsets):
            preds = tree.predict(X[:, feat_idx_subset])
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)

def cross_val_rmse(model_class, X, y, cv=5, random_state=42, **model_kwargs):
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=float)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_val_cv)

        mse = np.mean((y_val_cv - preds) ** 2)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    return np.array(rmses)

if __name__ == "__main__":

    y = df_possum["footlgth"].copy()
    X = df_possum.drop(columns=["footlgth"])

    cat_features = [1, 2]

    # 5-fold CV on the whole dataset
    cv_rmses = cross_val_rmse(
        RegRF,
        X,
        y,
        cv=5,
        random_state=42,
        n_estimators=50,
        max_depth=5,
        min_samples_split=2,
        cat_features=cat_features,
        max_features=None
    )

    print("CV RMSEs:", cv_rmses)
    print("Mean CV RMSE:", cv_rmses.mean())

    # Setup your independent and dependent variable
    y = df_possum["footlgth"].copy()
    X = df_possum.drop(columns=["footlgth"])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes:  X={X_test.shape},  y={y_test.shape}")

    # indices of categorical columns in X
    cat_features = [1, 2]

    rf = RegRF(
        n_estimators=50,
        max_depth=5,
        min_samples_split=2,
        cat_features=cat_features,
        max_features=5,    
        random_state=42
    )

    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    X_test_arr = np.array(X_test, dtype=object)

    # Predict the values
    predictions = rf.predict(X_test_arr)
    
    # Create result df
    results_df = pd.DataFrame(y_test).copy()
    results_df.columns = ["real_values"]
    results_df["predicted_values"] = predictions
    
    # Add Error column
    results_df["Error"] = results_df["real_values"] - results_df["predicted_values"]
    results_df["Squared_Error"] = results_df["Error"] ** 2

    # Inspect the results
    print(results_df.head(10))

    # Calculate overall performance
    mse = results_df["Squared_Error"].mean()
    rmse = np.sqrt(mse)
    print("Random Forest Test-RMSE:", rmse)