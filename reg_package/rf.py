
import numpy as np
from reg_package.tree import RegTree

class RegRF:
    """
    Random Forest Regressor implementation using bagging and feature subsetting.

    Builds an ensemble of regression trees where each tree is trained on a bootstrap 
    sample of the data and a random subset of features.
    """
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf = 0, cat_features=None, max_features=None, random_state=None):
        """
        Initializes the random forest.

        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            min_samples_split (int): Minimum samples required to split a node.
            min_samples_leaf (int): Minimum samples required at a leaf node.
            cat_features (list): Indices of categorical features.
            max_features (int): Number of features to consider when looking for the best split.
            random_state (int): Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cat_features = set(cat_features) if cat_features else set()
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []
        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        """Generates a bootstrap sample of rows."""
        n_samples = X.shape[0]
        indices = np.random.randint(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def _sample_features(self, n_features):
        """Selects a random subset of feature indices to be used for a specific tree."""
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return np.random.choice(n_features, self.max_features, replace=False)

    def fit(self, X, y):
        """
        Trains the random forest.

        For each estimator:
        1. Creates a bootstrap sample of the dataset.
        2. Selects a random subset of features.
        3. Trains a RegTree on this specific sample and feature subset.
        """
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
                           min_samples_leaf=self.min_samples_leaf,
                           cat_features=cat_in_subset)
            tree.fit(X_sample[:, feat_idx_subset], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Aggregates predictions from all trees.

        Args:
            X (array-like): Input data.

        Returns:
            np.array: The mean of predictions from all individual trees.
        """
        X = np.array(X, dtype=object)
        all_preds = []
        for tree, feat_idx_subset in zip(self.trees, self.feature_subsets):
            preds = tree.predict(X[:, feat_idx_subset])
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)