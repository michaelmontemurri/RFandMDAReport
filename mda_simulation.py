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

def generate_data(n_samples, n_features, c, tau_0):
    #generate the covariance matrix
    I = np.eye(n_features)
    ones = np.ones(n_features)
    C = (1-c)*I + c*np.outer(ones, ones)
    #make tau a column vector of tau_0 of lengeth p
    tau = [tau_0]*n_features

    # We need to make matrxi with C in the top left, and tau^T in the bottom left and tau in top right

    #create a 3x3 matrix with C in the top left, tau^T in the bottom left and tau in top right, and 1 in the bottom right
    C = np.block([[C, np.array(tau).reshape(-1, 1)], [np.array(tau).reshape(1, -1), 1]])
    print(C)

    #generate noise
    eps = np.random.normal(0, 1)

    #generate X
    mean = np.zeros(n_features+1)
    M = np.random.multivariate_normal(mean, C, n_samples)
    X = M[:, :-1]
    Y = M[:,-1] + eps

    return X, Y

#MDA result from Gregorutti (2016)
def mda_star(X, Y, tau_0, c, p):
    print(tau_0)
    theoretical_mda = 2 * (tau_0 / (1 - c + p * c)) ** 2 
    print(theoretical_mda)
    return theoretical_mda

#lets do a littel simulation to see if the result holds when we compare to the sklearn permutation importance

n_samples = 1000
n_features = 2
c = 0.01
tau_0 = 0.5
X,Y = generate_data(n_samples, n_features, c, tau_0)
print(X,Y)

#use generated data to fit a random forest model
rf = RandomForestRegressor(n_estimators=1000, max_depth=None, max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(X, Y)
perm_importance = permutation_importance(rf, X, Y, n_repeats=20, random_state=42)
importance = perm_importance.importances_mean


#now we want to comapre the result from the MDA formula to the permutation importance

#calculate MDA^*(X^(j)) for all j
theoretical_mda = mda_star(X, Y, tau_0, c, n_features)

print('theoretical mda', theoretical_mda)
print('sklearn perm. importatnce', importance)