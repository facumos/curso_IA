import numpy as np


from base_model import BaseModel, LinearRegression
from metric import Metric, MSE


def k_folds_poly(X_train, y_train, k=5, regression=LinearRegression()):

    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list_linear = []
    mse_list_quadratic = []
    mse_list_cubic = []
    mse_list_4 = []

    for j in range(0, 1, 1):
        for i in range(0, len(X_train), chunk_size):
            end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
            new_X_valid = X_train[i: end]
            new_y_valid = y_train[i: end]
            new_X_train = np.concatenate([X_train[: i], X_train[end:]])
            new_y_train = np.concatenate([y_train[: i], y_train[end:]])

            # linear
            X_linear = np.vstack((np.power(new_X_train, 1), np.power(new_X_train, 0))).T
            regression.fit(X_linear, new_y_train.reshape(-1, 1))
            W_linear = regression.model
            y_linear = W_linear[0]*new_X_valid + W_linear[1]

            mse_list_linear.append(error(new_y_valid, y_linear))


        # quadratic
        X_quadratic = np.vstack((np.power(new_X_train, 2), np.power(new_X_train, 1), np.power(new_X_train, 0))).T
        regression.fit(X_quadratic, new_y_train.reshape(-1, 1))
        W_quadratic = regression.model
        y_quadratic = W_quadratic[0] * np.power(new_X_valid, 2) + W_quadratic[1] * np.power(new_X_valid, 1)\
                    + W_quadratic[2] * np.power(new_X_valid, 0)

        mse_list_quadratic.append(error(new_y_valid, y_quadratic))

        # cubic
        X_cubic = np.vstack((np.power(new_X_train, 3), np.power(new_X_train, 2), new_X_train,
                             np.ones(len(new_X_train)))).T
        regression.fit(X_cubic, new_y_train.reshape(-1, 1))
        W_cubic = regression.model
        y_cubic = W_cubic[0] * np.power(new_X_valid, 3) + W_cubic[1] * np.power(new_X_valid, 2) + \
                  W_cubic[2] * new_X_valid + W_cubic[3]

        mse_list_cubic.append(error(new_y_valid, y_cubic))

        # X4
        X_4 = np.vstack((np.power(new_X_train, 4), np.power(new_X_train, 3), np.power(new_X_train, 2), new_X_train,
                             np.ones(len(new_X_train)))).T
        regression.fit(X_4, new_y_train.reshape(-1, 1))
        W_4 = regression.model
        y_4 = W_4[0] * np.power(new_X_valid, 4) + W_4[1] * np.power(new_X_valid, 3) + \
              W_4[2] * np.power(new_X_valid, 2) + W_4[3] * new_X_valid + W_4[4]

        mse_list_4.append(error(new_y_valid, y_4))


    mean_MSE_linear = np.mean(mse_list_linear)
    mean_MSE_quadratic = np.mean(mse_list_quadratic)
    mean_MSE_cubic = np.mean(mse_list_cubic)
    mean_MSE_4 = np.mean(mse_list_4)

    return mean_MSE_linear, mean_MSE_quadratic, mean_MSE_cubic, mean_MSE_4
