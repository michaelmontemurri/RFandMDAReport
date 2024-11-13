import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from randomforest import RandomForest, RandomForestParallel, PurelyRandomForest
from sklearn.ensemble import RandomForestRegressor

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

    # custom_model = RandomForest(n_trees=50, max_depth=10)
    # custom_model.fit(X_train, y_train)
    # y_pred_custom = custom_model.predict(X_test)
    # rmse_custom = np.sqrt(np.mean((y_test - y_pred_custom) ** 2))
    # print(f"RMSE custom: {rmse_custom}")
    # nrmse_custom = rmse_custom / (y_test.max() - y_test.min())
    # print(f"NRMSE custom: {nrmse_custom}")

    # fit sklearn model

    sklearn_model = RandomForestRegressor(n_estimators=100, max_depth=1000, n_jobs=-1)
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    rmse_sklearn = np.sqrt(np.mean((y_test - y_pred_sklearn) ** 2))
    print(f"RMSE sklearn: {rmse_sklearn}")
    nrmse_sklearn = rmse_sklearn / (y_test.max() - y_test.min())
    print(f"NRMSE sklearn: {nrmse_sklearn}")

    # fit parallel rf model
    # parallel_model = RandomForestParallel(n_trees=100, max_depth=50)
    # parallel_model.fit(X_train, y_train)
    # y_pred_parallel = parallel_model.predict(X_test)
    # rmse_parallel = np.sqrt(np.mean((y_test - y_pred_parallel) ** 2))
    # print(f"RMSE parallel: {rmse_parallel}")
    # nrmse_parallel = rmse_parallel / (y_test.max() - y_test.min())
    # print(f"NRMSE parallel: {nrmse_parallel}")

    #fit purely random forest model
    purely_random_model = PurelyRandomForest(n_trees=100, k=50)
    purely_random_model.fit(X_train, y_train)
    y_pred_purely_random = purely_random_model.predict(X_test)
    rmse_purely_random = np.sqrt(np.mean((y_test - y_pred_purely_random) ** 2))
    print(f"RMSE purely random: {rmse_purely_random}")
    nrmse_purely_random = rmse_purely_random / (y_test.max() - y_test.min())
    print(f"NRMSE purely random: {nrmse_purely_random}")



    # # plot results
    # plt.figure(figsize=(12, 5))

    # # custom model predictions
    # plt.subplot(1, 2, 1)
    # plt.scatter(y_test, y_pred_custom, alpha=0.6)
    # plt.xlabel("True Price")
    # plt.ylabel("Predicted Price")
    # plt.title("Custom Model: True Price vs Predicted Price")

    # # sklearn model predictions
    # plt.subplot(1, 2, 2)
    # plt.scatter(y_test, y_pred_sklearn, alpha=0.6, color='orange')
    # plt.xlabel("True Price")
    # plt.ylabel("Predicted Price")
    # plt.title("Sklearn Model: True Price vs Predicted Price")

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
