import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
#Let's start by simulating data that follows the constraints required by the result  in Gregorutti (2016)
#  consider model of the form Y=m(X)+eps, wheere (X, eps_ is a Gaussian random vector
# assume the correlation matrix satsifies the following conditions:
#C = [Cov(X^(j), X^(k)] = (1-c)I_p + c11^T, where c is a constant in [0, 1], I_p is the p-dimensional identity matrix, and 1 is a vector of ones)]
#also assum Cov(X^(j), Y) = tau_0 for all j in {1, ..., p}
#rseult says MDA^*(X^(j)) = 2(tau_0/(1-c+pc))^2
#i.e in a gaussian setting variable importance decreases as the inverse of the square of p when the number of correlated variable p increase.

#lets write a function to do this

def generate_data(n_samples, n_features, c, beta):
    I = np.eye(n_features)
    ones = np.ones(n_features)
    C = (1-c)*I + c*np.outer(ones, ones)

    #check if C is positive definite
    assert np.all(np.linalg.eigvals(C) > 0)

    #generate X
    mean = np.zeros(n_features)
    X = np.random.multivariate_normal(mean, C, n_samples)

    #generate noise
    eps = np.random.normal(0, 1, n_samples)
    #get Y
    Y = X @ beta + eps
    return X, Y

#MDA result from Gregorutti (2016)
def mda_star_per_feature(X, Y, c, p):
    tau_0s = [np.corrcoef(X[:, j], Y)[0, 1] for j in range(X.shape[1])]
    print(tau_0s)
    theoretical_mda = [2 * (tau_0 / (1 - c + p * c)) ** 2 for tau_0 in tau_0s]
    return theoretical_mda

#lets do a littel simulation to see if the result holds when we compare to the sklearn permutation importance

n_samples = 10
n_features = 5
c = 0.5
beta = np.random.normal(0, 1, n_features)
X,Y = generate_data(n_samples, n_features, c, beta)

#use generated data to fit a random forest model
rf = RandomForestRegressor(n_estimators=1000, max_depth=None, max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(X, Y)
perm_importance = permutation_importance(rf, X, Y, n_repeats=20, random_state=42)
importance = perm_importance.importances_mean


#now we want to comapre the result from the MDA formula to the permutation importance

#calculate MDA^*(X^(j)) for all j
theoretical_mda = mda_star_per_feature(X, Y, c, n_features)

for j, (imp, mda) in enumerate(zip(importance, theoretical_mda)):
    print(f"Feature {j+1}: permutation importance(MDA) = {imp:.4f}, theoretical MDA* = {mda:.4f}")
