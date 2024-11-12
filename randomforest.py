import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    def grow_tree(self, X, y, depth=0):
        min_var = 1e-7
        # Check if we are at a leaf node or max depth
        if len(set(y)) <= 1 or depth >= self.max_depth:
            self.value = np.mean(y)
            return
        
        # randomly select features
        num_features = int(np.sqrt(X.shape[1]))
        random_features = np.random.choice(X.shape[1], num_features, replace=False)
 
        # get best split criteria
        feature, threshold = self._find_split(X, y, random_features)

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
        self.left.grow_tree(X[left_idxs], y[left_idxs], depth + 1)

        self.right = DecisionTree(self.max_depth)
        self.right.grow_tree(X[right_idxs], y[right_idxs], depth + 1)



    def _get_thresholds(self, feature_values):
        """
        Get possible split thresholds for the values of a feature by finding midpoints
        """
        feature_values = np.sort(feature_values)
        midpoints = (feature_values[1:] + feature_values[:-1]) / 2
        return midpoints

    def _find_split(self, X, y, random_features):
        """
        Find the best split for the data given features
        """
        #set minimum var threshold
        if np.var(y) < 1e-7:
            return None, None

        # initialize
        best_split = None
        best_mse = float('inf')

        #loop through selected features and thresholds to find best split
        for feature in random_features:
            feature_values = X[:, feature]
            thresholds = self._get_thresholds(feature_values)

            for threshold in thresholds:
                left_idxs = feature_values < threshold
                right_idxs = feature_values >= threshold

                y_left, y_right = y[left_idxs], y[right_idxs]

                # skip splits where either left or right partition is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # calculate mse of split 
                mse = (np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)) / (len(y_left) + len(y_right))

                #update best mse and split
                if mse < best_mse:
                    best_mse = mse
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
