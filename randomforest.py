import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed, cpu_count

def bootstrap_sample(X, y):
    """
    Create a bootstrap sample of X and y
    :param X: np.array, shape (n_samples, n_features)
    :param y: np.array, shape (n_samples,)
    :return: X_bootstrap, y_bootstrap: np.arrays
    """
    # get number of samples
    n_samples = X.shape[0]

    # get random indices (with replacement) for bootstraped sample
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)

    # return only the samples with the selected indices
    return X[idxs], y[idxs]


class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None  # For storing leaf values
    
    def fit(self, X, y):
        """
        Fit a fully grown decision tree to the data
        """
        self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0, min_var=1e-7, subset_features=False):
   
        # Check if we are at a leaf node or max depth
        if len(set(y)) <= 1  or np.var(y) < min_var or depth >= self.max_depth:
            self.value = np.mean(y)
            return
        
        #if subset features is true, randomly select features, otherwise use all features (typically use all features for regression)
        if subset_features:
            # randomly select features
            num_features = int(np.sqrt(X.shape[1]))
            features = np.random.choice(X.shape[1], num_features, replace=False)
        else:
            # use all features
            features = np.arange(X.shape[1])
 
        # get best split criteria
        feature, threshold = self._find_split(X, y, features)

        # check if split is valid
        if feature is None:
            self.value = np.mean(y)
            return
        
        # store split criteria for this node
        self.feature = feature
        self.threshold = threshold

        # make the split and grow children recursively
        left_idxs = X[:, feature] < threshold
        right_idxs = X[:, feature] >= threshold

        self.left = DecisionTree(self.max_depth)
        self.left.grow_tree(X[left_idxs], y[left_idxs], depth + 1, min_var, subset_features)

        self.right = DecisionTree(self.max_depth)
        self.right.grow_tree(X[right_idxs], y[right_idxs], depth + 1, min_var, subset_features)

    def _get_thresholds(self, feature_values):
        """
        Get possible split thresholds for the values of a feature by finding midpoints
        """
        feature_values = np.sort(feature_values)
        midpoints = (feature_values[1:] + feature_values[:-1]) / 2
        return midpoints

    def _find_split(self, X, y, features):
        """
        Find the best split for the data given features
        """
        #set minimum var threshold
        if np.var(y) < 1e-7:
            return None, None

        # initialize
        best_split = None
        best_mse = float('-inf')

        #loop through selected features and thresholds to find best split
        for feature in features:
            feature_values = X[:, feature]
            thresholds = self._get_thresholds(feature_values)

            for threshold in thresholds:
                left_idxs = feature_values < threshold
                right_idxs = feature_values >= threshold

                y_left, y_right = y[left_idxs], y[right_idxs]

                # skip splits where either left or right partition is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue


                #calcualte mse of children
                n_parent = len(y)
                n_left = len(y_left)
                n_right = len(y_right)

                var_parent = np.var(y)
                var_left = np.var(y_left)
                var_right = np.var(y_right)

                weighted_left = (n_left / n_parent) * var_left 
                weighted_right = (n_right / n_parent) * var_right

                mse_decrease = var_parent - (weighted_left + weighted_right)


                #update best mse and split
                if mse_decrease > best_mse:
                    best_mse = mse_decrease
                    best_split = (feature, threshold)
        
        return best_split if best_split is not None else (None, None)
    
    def _predict_row(self, x):
        """
        Predict a single row of data
        """
        #cehck if leaf node
        if self.value is not None:
            return self.value
        # otherwise, compare feature to threshold and recurse
        if x[self.feature] < self.threshold:
            return self.left._predict_row(x)
        else:
            return self.right._predict_row(x)
        
    def predict(self, X):
        """
        Predict the entire dataset
        """
        
        return np.array([self._predict_row(x) for x in X])
    
  
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        """
        Fit a random forest to the data
        """
        # loop through number of trees and fit them
        for _ in range(self.n_trees):
            # create decision tree
            tree = DecisionTree(max_depth=self.max_depth)

            #get bootstrap sample
            X_bootstrap, y_bootstrap = bootstrap_sample(X, y)

            #fit tree
            tree.fit(X_bootstrap, y_bootstrap)

            #append tree to list
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Predict the data using the fitted random forest
        """
        #just get the average prediction across all trees
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


class RandomForestParallel:
    def __init__(self, n_trees=100, max_depth=10, n_jobs=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()  # default to all cores if -1
        self.trees = []
        
        print(f"RandomForestParallel initialized with {self.n_jobs} cores out of {cpu_count()} available.")
        
    def _fit_tree(self, X, y):
        X_bootstrap, y_bootstrap = bootstrap_sample(X, y)
        tree = DecisionTree(max_depth=self.max_depth)
        tree.fit(X_bootstrap, y_bootstrap)
        return tree
    
    def fit(self, X, y):
        """
        Fit a random forest to the data using parallel processing.
        """
        # fit each tree in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_tree)(X, y) for _ in range(self.n_trees)
        )
    
    def predict(self, X):
        """
        Predict the data using the fitted random forest in parallel.
        """
        # collect predictions from each tree in parallel
        tree_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        
        # avg prdeicitons across all trees
        return np.mean(tree_preds, axis=0)
    

class PurelyRandomDecisionTree:
    def __init__(self, k=10):
        self.k = k #max depth for purely random
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None  # For storing leaf values
    
    def fit(self, X, y):
        """
        Fit a fully grown decision tree to the data
        """
        self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0, min_var=1e-7):
        
        #stop if we reach max depth or if we have a pure node
        if depth >= self.k or len(y) == 1:
            self.value = np.mean(y)
            return
        
        # choose feature uniformly at random
        feature = np.random.choice(X.shape[1])
 
        valid_rows = np.isfinite(X[:, feature])  # true for rows without NaN or inf
        X_valid = X[valid_rows]
        y_valid = y[valid_rows]
        
        if len(X_valid) == 0:  # if no valid rows, we can't split
            self.value = np.nanmean(y_valid)  # store the mean as the leaf value
            return
        
        # split on the center of the cell on the chosen feature
        threshold = np.nanmean(X_valid[:, feature])
        
        # store split criteria for this node
        self.feature = feature
        self.threshold = threshold

        # make the split and grow children recursively
        left_idxs = X_valid[:, feature] < threshold
        right_idxs = X_valid[:, feature] >= threshold

        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            self.value = np.mean(y_valid)  # Use the mean of valid data as the leaf value
            return
        
        self.left = PurelyRandomDecisionTree(self.k)
        self.left.grow_tree(X_valid[left_idxs], y_valid[left_idxs], depth + 1, min_var)

        self.right = DecisionTree(self.k)
        self.right.grow_tree(X_valid[right_idxs], y_valid[right_idxs], depth + 1, min_var)


    def _predict_row(self, x):
        """
        Predict a single row of data
        """
        #cehck if leaf node
        if self.value is not None:
            return self.value
        # otherwise, compare feature to threshold and recurse
        if x[self.feature] < self.threshold:
            return self.left._predict_row(x)
        else:
            return self.right._predict_row(x)
        
    def predict(self, X):
        """
        Predict the entire dataset
        """
        
        return np.array([self._predict_row(x) for x in X])

class PurelyRandomForest:
    def __init__(self, n_trees=100, k=10, n_jobs=-1):
        self.n_trees = n_trees
        self.k = k
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()  # default to all cores if -1
        self.trees = []
        
        print(f"RandomForestParallel initialized with {self.n_jobs} cores out of {cpu_count()} available.")
            
    def _fit_tree(self, X, y):
        tree = PurelyRandomDecisionTree(k=self.k)
        tree.fit(X, y)
        return tree
    
    def fit(self, X, y):
        """
        Fit a random forest to the data using parallel processing.
        """
        # fit each tree in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_tree)(X, y) for _ in range(self.n_trees)
        )
    
    def predict(self, X):
        """
        Predict the data using the fitted random forest in parallel.
        """
        # collect predictions from each tree in parallel
        tree_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        
        # avg prdeicitons across all trees
        return np.mean(tree_preds, axis=0)