import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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


    #generate noise
    eps = np.random.normal(0, 1)

    #generate X
    mean = np.zeros(n_features+1)
    M = np.random.multivariate_normal(mean, C, n_samples)
    X = M[:, :-1]
    Y = M[:,-1] + eps

    return X, Y

#Theoretical MDA result from Gregorutti (2016)
def mda_star(tau_0, c, p):
 
    theoretical_mda = 2 * (tau_0 / (1 - c + p * c)) ** 2 
    return theoretical_mda


def simulation(n_samples, p, c, tau_0): 

    X,Y = generate_data(n_samples, p, c, tau_0)

    rf.fit(X, Y)

    perm_importance = permutation_importance(rf, X, Y, n_repeats=20, random_state=42)

    importance = perm_importance.importances_mean
    #scale importances by times by 2
    importance = importance

    theoretical_mda_val = mda_star(tau_0, c, p)

    # store the results
    sklearn_mda[(p, c)] = np.mean(importance)
    theoretical_mda[(p, c)] = theoretical_mda_val

    return sklearn_mda, theoretical_mda

#lets do a littel simulation to see if the result holds when we compare to the sklearn permutation importance

n_samples = 10000
rf = RandomForestRegressor(n_estimators=100, max_depth=None, max_features='sqrt', n_jobs=-1)

#initialize dictionary to store results
sklearn_mda = {}
theoretical_mda = {}

#define valid maxmim values for tau_0 for each p, 
# we need C to be positive semi-definite, so all eigen values of C must be positive. This puts a tighter restriction on tau_0 (depending on p) than was mentioned in the proof. We believe this is why the emperical results are not matching the theoretical results
p_values = [3]
c_values = [.1,.2,.3,.4, .5, .6,.7,.8, .9]
tau_0 = .2

for p in p_values:
    for c in c_values:
        sklearn_mdas, theoretical_mdas, = simulation(n_samples, p, c, tau_0)

print('sklearn_mdas', sklearn_mdas)
print('theoretical_mdas', theoretical_mdas)

# Initialize plot
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot for sklearn_mdas
for p in p_values:
    # Extract the c values and corresponding MDA values for each p
    c_vals = [c for c in c_values]
    mda_vals = [sklearn_mdas[(p, c)] for c in c_vals]
    
    axes[0].plot(c_vals, mda_vals, label=f"p = {p}")

axes[0].set_title("Sklearn MDA vs c values")
axes[0].set_xlabel("c value")
axes[0].set_ylabel("MDA (Sklearn)")
axes[0].legend()

# Plot for theoretical_mdas
for p in p_values:
    # Extract the c values and corresponding theoretical MDA values for each p
    c_vals = [c for c in c_values]
    mda_vals = [theoretical_mdas[(p, c)] for c in c_vals]
    
    axes[1].plot(c_vals, mda_vals, label=f"p = {p}")

axes[1].set_title("Theoretical MDA vs c values")
axes[1].set_xlabel("c value")
axes[1].set_ylabel("MDA (Theoretical)")
axes[1].legend()

# Show the plot
plt.tight_layout()
plt.show()