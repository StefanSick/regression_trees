import pandas as pd
import numpy as np
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
    """
    Regression Tree implementation using Mean Squared Error (MSE) reduction.

    Supports both numerical (<= split) and categorical (== split) features.
    """
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, cat_features=None):
        """
        Args:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum samples required to split a node.
            min_samples_leaf (int): Minimum samples required at a leaf node.
            cat_features (list): Indices of categorical features (uses equality splitting).
        """
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.cat_features = set(cat_features) if cat_features else set() 
        self.root = None

    def fit(self, X, y):
        """
        Builds the regression tree using the training data.

        Recursively splits data to minimize variance until stopping criteria (depth, 
        min samples, or pure node) are met.
        """
        X = np.array(X, dtype=object) 
        y = np.array(y, dtype=float)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively builds tree nodes by finding the best split at each step."""
        n_samples, n_features = X.shape

        # If no samples, just return a leaf with value 0 (or some default)
        if n_samples == 0:
            return Node(value=0.0)

        # Stopping Criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.std(y) < 1e-8):
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
        """
        Iterates through features and thresholds to find the split maximizing MSE reduction.
        
        Note: Uses percentiles [25, 50, 75] as thresholds if a feature has >100 unique values 
        to speed up training.
        """
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
        """Calculates the reduction in MSE for a potential split."""
        left_idxs, right_idxs = self._split(X_column, threshold, feat_idx)

        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
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
        """
        Splits data indices based on feature type.
        
        Returns:
            left_idxs, right_idxs: Indices for children nodes. 
            Uses (val <= thresh) for numerical and (val == thresh) for categorical.
        """
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
            