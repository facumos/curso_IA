import numpy as np
import matplotlib.pyplot as plt


from data import Data

from base_model import BaseModel, ConstantModel, LinearRegression, LinearRegressionWithB
from metric import Metric, MSE


def k_folds_poly(X_train, y_train, k=5, regression=LinearRegression()):

    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list_linear = []
    mse_list_quadratic = []
    mse_list_cubic = []
    mse_list_4 = []
    mse_list_5 = []
    mse_list_6 = []
    mse_list_7 = []
    mse_list_9 = []
    mse_list_10 = []

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

        # X5
        X_5 = np.vstack((np.power(new_X_train, 5), np.power(new_X_train, 4), np.power(new_X_train, 3),
                         np.power(new_X_train, 2), new_X_train, np.ones(len(new_X_train)))).T
        regression.fit(X_5, new_y_train.reshape(-1, 1))
        W_5 = regression.model
        y_5 = W_5[0] * np.power(new_X_valid, 5) + W_5[1] * np.power(new_X_valid, 4) + \
              W_5[2] * np.power(new_X_valid, 3) + W_5[3] * np.power(new_X_valid, 2) + W_5[4] * new_X_valid + W_5[5]

        mse_list_5.append(error(new_y_valid, y_5))

        # X6
        X_6 = np.vstack((np.power(new_X_train, 6), np.power(new_X_train, 5), np.power(new_X_train, 4),
                         np.power(new_X_train, 3), np.power(new_X_train, 2), new_X_train, np.ones(len(new_X_train)))).T
        regression.fit(X_6, new_y_train.reshape(-1, 1))
        W_6 = regression.model
        y_6 = W_6[0] * np.power(new_X_valid, 6) + W_6[1] * np.power(new_X_valid, 5) + \
              W_6[2] * np.power(new_X_valid, 4) + W_6[3] * np.power(new_X_valid, 3) + \
              W_6[4] * np.power(new_X_valid, 2) + W_6[5] * np.power(new_X_valid, 1) + W_6[6]

        mse_list_6.append(error(new_y_valid, y_6))

        # X7
        X_7 = np.vstack((np.power(new_X_train, 7), np.power(new_X_train, 6), np.power(new_X_train, 5),
                         np.power(new_X_train, 4), np.power(new_X_train, 3), np.power(new_X_train, 2),
                         new_X_train, np.ones(len(new_X_train)))).T
        regression.fit(X_7, new_y_train.reshape(-1, 1))
        W_7 = regression.model
        y_7 = W_7[0] * np.power(new_X_valid, 7) + W_7[1] * np.power(new_X_valid, 6) + \
              W_7[2] * np.power(new_X_valid, 5) + W_7[3] * np.power(new_X_valid, 4) + \
              W_7[4] * np.power(new_X_valid, 3) + W_7[5] * np.power(new_X_valid, 2) + \
              W_7[6] * np.power(new_X_valid, 1) + W_7[7] * np.power(new_X_valid, 0)

        mse_list_7.append(error(new_y_valid, y_7))

        # X9
        X_9 = np.vstack((np.power(new_X_train, 9), np.power(new_X_train, 8), np.power(new_X_train, 7),
                         np.power(new_X_train, 6), np.power(new_X_train, 5),
                         np.power(new_X_train, 4), np.power(new_X_train, 3), np.power(new_X_train, 2),
                         new_X_train, np.ones(len(new_X_train)))).T
        regression.fit(X_9, new_y_train.reshape(-1, 1))
        W_9 = regression.model
        y_9 = W_9[0] * np.power(new_X_valid, 9) + W_9[1] * np.power(new_X_valid, 8) + \
              W_9[2] * np.power(new_X_valid, 7) + W_9[3] * np.power(new_X_valid, 6) + \
              W_9[4] * np.power(new_X_valid, 5) + W_9[5] * np.power(new_X_valid, 4) + \
              W_9[6] * np.power(new_X_valid, 3) + W_9[7] * np.power(new_X_valid, 2) + \
              W_9[8] * np.power(new_X_valid, 1) + W_9[9] * np.power(new_X_valid, 0)

        mse_list_9.append(error(new_y_valid, y_9))

        # X10
        X_10 = np.vstack((np.power(new_X_train, 10), np.power(new_X_train, 9), np.power(new_X_train, 8),
                        np.power(new_X_train, 7), np.power(new_X_train, 6), np.power(new_X_train, 5),
                        np.power(new_X_train, 4), np.power(new_X_train, 3), np.power(new_X_train, 2),
                        new_X_train, np.ones(len(new_X_train)))).T
        regression.fit(X_10, new_y_train.reshape(-1, 1))
        W_10 = regression.model
        y_10 = W_10[0] * np.power(new_X_valid, 10) + W_10[1] * np.power(new_X_valid, 9) + \
               W_10[2] * np.power(new_X_valid, 8) + W_10[3] * np.power(new_X_valid, 7) + \
               W_10[4] * np.power(new_X_valid, 6) + W_10[5] * np.power(new_X_valid, 5) + \
               W_10[6] * np.power(new_X_valid, 4) + W_10[7] * np.power(new_X_valid, 3) + \
               W_10[8] * np.power(new_X_valid, 2) + W_10[9] * new_X_valid + W_10[10]

        mse_list_10.append(error(new_y_valid, y_10))

    mean_MSE_linear = np.mean(mse_list_linear)
    mean_MSE_quadratic = np.mean(mse_list_quadratic)
    mean_MSE_cubic = np.mean(mse_list_cubic)
    mean_MSE_4 = np.mean(mse_list_4)
    mean_MSE_5 = np.mean(mse_list_5)
    mean_MSE_6 = np.mean(mse_list_6)
    mean_MSE_7 = np.mean(mse_list_7)
    mean_MSE_9 = np.mean(mse_list_9)
    mean_MSE_10 = np.mean(mse_list_10)

    # PLOTS
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('Sin(x) - Fitting curves')

    x = new_X_valid
    # noisy_y = new_y_train
    # original
    plt.plot(new_X_valid, new_y_valid, 'o')

    # linear
    plt.plot(x, y_linear, 'o')

    # quadratic
    plt.plot(x, y_quadratic, 'o')

    # cubic
    plt.plot(x, y_cubic, 'o')

    # 4 power
    plt.plot(x, y_4, 'o')

    # 5 power
    plt.plot(x, y_5, 'o')

    # 6 power
    plt.plot(x, y_6, 'o')

    # # 7 power
    plt.plot(x, y_7, 'o')

    # # 9 power
    plt.plot(x, y_9, 'o')

    # # 10 power
    plt.plot(x, y_10, 'o')

    plt.legend(['noisy signal', 'linear', 'quadratic', 'cubic', '4th power', '5th power', '6th power',
                '7th power', '9th power', '10th power'])
    plt.show()

    return mean_MSE_linear, mean_MSE_quadratic, mean_MSE_cubic, mean_MSE_4, mean_MSE_5,\
           mean_MSE_6, mean_MSE_7, mean_MSE_9, mean_MSE_10
