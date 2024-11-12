import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from randomforest import RandomForest

#test implementation of random forest code
def main(): 
    # load toy dataset
    data = datasets.fetch_california_housing()

    #just use first 1000 samples for speed
    X = data.data[:1000]  
    y = data.target[:1000]      
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit rf model
    model = RandomForest(n_trees=50, max_depth=10)
    model.fit(X_train, y_train)

    # predict on the test data
    y_pred = model.predict(X_test)

    # calculate pred accuracy
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"RMSE: {rmse}")
    nrmse = rmse / (y_test.max() - y_test.min())
    print(f"NRMSE: {nrmse}")

    # plot results
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("True Price vs Predicted Price")
    plt.show()


if __name__ == "__main__":
    main()
