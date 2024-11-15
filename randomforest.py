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

                #calcualte mse decrease of split
                mse_decrease = self._information_gain(y, y_left, y_right)

                #update best mse and split
                if mse_decrease > best_mse:
                    best_mse = mse_decrease
                    best_split = (feature, threshold)
        
        return best_split if best_split is not None else (None, None)
    

    def _information_gain(self, y, y_left, y_right):
        """
        Calculate the information gain of a split
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        mse_parent = np.var(y) * n
        mse_left = np.var(y_left) * n_left
        mse_right = np.var(y_right) * n_right

        weighted_mse_children = (mse_left + mse_right) / n
        return mse_parent - weighted_mse_children
    
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

        self.right = PurelyRandomDecisionTree(self.k)
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
    
class TotallyRandomDecisionTree:
    def __init__(self):
        self.children = {} #dictionary to store children based on modalities
        self.value = None  # For storing leaf values
        self.feature = None
        self.mse_decrease = {}

    def fit(self, X, y):
        """
        Fit a fully grown decision tree to the data
        """
        self.grow_tree(X, y, depth=0, available_features = list(range(X.shape[1])))

    def grow_tree(self, X, y, depth=0, available_features = None):
        # we need to store the impurity decrease associated with each split and its corresponding feature so that we can calculate feature importance later

        #in this framework, if we are at depth p, we are at a leaf node
        if len(available_features) == 0:
            self.value = np.mean(y) 
            return
 
        #choose random feature to split on without replacement, i.e. once a feature is chosen, it cannot be chosen again in this tree.
        self.feature = np.random.choice(available_features)
        
        #update remaining features, could be a faster way to do this but ok for now
        new_features = [f for f in available_features if f != self.feature]

        #get modalities of feature (number of unique values in the feature)
        modalities = np.unique(X[:, self.feature])
        
        total_mse_decrease = 0
        #split the data according to the modality
        for mod in modalities:
            #get indices of data where the feature is equal to the modality
            idxs = X[:, self.feature] == mod
            X_mod, y_mod = X[idxs], y[idxs]
            
            #if the modality is not present in the data, we need to skip the split
            if len(y_mod) == 0:
                continue

            #measure information gain from split, in this case we have multipkle children per node, so we need to sum the impurity decrease over all children
            mse_decrease = self._information_gain(y, y_mod, y[~idxs])
            total_mse_decrease += mse_decrease

            #store impurity decrease for this specific split and the correpsonding feature
            if self.feature not in self.mse_decrease:
                self.mse_decrease[self.feature] = 0
            self.mse_decrease[self.feature] += mse_decrease

            #create a child node and grow tree recursively
            child = TotallyRandomDecisionTree()
            child.grow_tree(X_mod, y_mod, depth + 1, new_features)
            self.children[mod] = child

        #now, calculate the average impurity decrease for each feature if feature importance is needed (to normalize again or already normazlized?)
        #actually in this case, each feature gets split on only once per tree, so we can just take the average of the impurity decrease over all trees, so we dont need to normalize
        #we can just sum the impurity decrease over all trees and then divide by the number of trees to get the average impurity decrease
    
    def _predict_row(self, x):
            """
            Predict a single row of data
            """
            #check if leaf node
            if self.value is not None:
                return self.value
            
            modality = x[self.feature]
           
            if modality in self.children:
                return self.children[modality]._predict_row(x)
            else:
                return self.value

    def predict(self, X):
        """
        Predict the entire dataset
        """
        return np.array([self._predict_row(x) for x in X])
    
    def _information_gain(self, y, y_left, y_right):
        """
        Calculate the information gain of a split
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        mse_parent = np.var(y) * n
        mse_left = np.var(y_left) * n_left
        mse_right = np.var(y_right) * n_right

        weighted_mse_children = (mse_left + mse_right) / n
        return mse_parent - weighted_mse_children

class TotallyRandomForest:
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        """
        Fit a random forest to the data.
        """
        for _ in range(self.n_trees):
            # Create and fit each tree on the data
            tree = TotallyRandomDecisionTree()
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the data using the fitted random forest by averaging predictions from each tree.
        """
        # Collect predictions from each tree
        # tree_preds = []
        # for tree in self.trees:
        #     pred = tree.predict(X)
        #     if pred is not None:
        #         tree_preds.append(pred)
        #     else:
        #         print('Warning: encountered NaN in prediction. Skipping tree.')
        
        # if not tree_preds:
        #     raise ValueError('All trees produced NaN predictions. Cannot make ensemble prediction.')

        # tree_preds = np.array(tree_preds)
        # Average predictions across trees
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)
    
    # def fit(self, X, y):
    #     """
    #     Fit a random forest to the data using parallel processing.
    #     """
    #     # fit each tree in parallel
    #     self.trees = Parallel(n_jobs=self.n_jobs)(
    #         delayed(self._fit_tree)(X, y) for _ in range(self.n_trees)
    #     )
    
    # def predict(self, X):
    #     """
    #     Predict the data using the fitted random forest in parallel.
    #     """
    #     # collect predictions from each tree in parallel
    #     tree_preds = np.array([tree.predict(X) for tree in self.trees])

    #     # 
    #     if tree_preds.ndim == 2 and tree_preds.shape[1] == 1: 
    #         return np.mean(tree_preds, axis=0)
    
    def feature_importances(self):
        """
        calculate feature importance
        """

        #initialize dictionary to store impurity decrease
        total_mse_decrease = {}

        #accumulate the mse_decrease with each corresponding feature over all trees
        for tree in self.trees:
            for feature, mse_decrease in tree.mse_decrease.items():
                if feature not in total_mse_decrease:
                    total_mse_decrease[feature] = 0
                total_mse_decrease[feature] += mse_decrease
        
        #normalize by the total impurity decrease
        total_decrease = sum(total_mse_decrease.values())

        #avoid division by zero for pure nodes
        if total_decrease == 0:
            return {feature: 0 for feature in total_mse_decrease.keys()}
        
        feature_importances = {feature: mse_decrease / total_decrease for feature, mse_decrease in total_mse_decrease.items()}

        return feature_importances
