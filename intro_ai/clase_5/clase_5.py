import time

import numpy as np
import matplotlib.pyplot as plt


from data import Data

from base_model import BaseModel, ConstantModel, LinearRegression, LinearRegressionWithB
from base_model import GradientDescent, StochasticGradientDescent, MiniBatchGradientDescent

from metric import Metric, MSE


def k_folds(X_train, y_train, k=5, ajuste=LinearRegression()):
    # l_regression = LinearRegression()
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        ajuste.fit(new_X_train, new_y_train)
        prediction = ajuste.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    mean_MSE = np.mean(mse_list)

    return mean_MSE


def sin_fitting_example():
    # y = sin(x)
    amt_points = 36
    x = np.linspace(0, 360, num=amt_points)
    y = np.sin(x * np.pi / 180.)
    noise = np.random.normal(0, .1, y.shape)
    noisy_y = y + noise

    X_train = x
    y_train = noisy_y

    regression = LinearRegression()

    # linear
    X_linear = np.vstack((X_train, np.ones(len(X_train)))).T
    regression.fit(X_linear, y_train.reshape(-1, 1))
    W_linear = regression.model
    y_linear = W_linear[0]*x + W_linear[1]

    # quadratic
    X_quadratic = np.vstack((np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regression.fit(X_quadratic, y_train.reshape(-1, 1))
    W_quadratic = regression.model
    y_quadratic = W_quadratic[0] * np.power(x, 2) + W_quadratic[1] * x + W_quadratic[2]

    # cubic
    X_cubic = np.vstack((np.power(X_train, 3), np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regression.fit(X_cubic, y_train.reshape(-1, 1))
    W_cubic = regression.model
    y_cubic = W_cubic[0] * np.power(x, 3) + W_cubic[1] * np.power(x, 2) + W_cubic[2] * x + W_cubic[3]

    # X10
    X_10 = np.vstack((np.power(X_train, 10), np.power(X_train, 9), np.power(X_train, 8),
                      np.power(X_train, 7), np.power(X_train, 6), np.power(X_train, 5),
                      np.power(X_train, 4), np.power(X_train, 3), np.power(X_train, 2),
                      X_train, np.ones(len(X_train)))).T
    regression.fit(X_10, y_train.reshape(-1, 1))
    W_10 = regression.model
    y_10 = W_10[0] * np.power(x, 10) + W_10[1] * np.power(x, 9) + W_10[2] * np.power(x, 8) + \
        W_10[3] * np.power(x, 7) + W_10[4] * np.power(x, 6) + W_10[5] * np.power(x, 5) + \
        W_10[6] * np.power(x, 4) + W_10[7] * np.power(x, 3) + W_10[8] * np.power(x, 2) + \
        W_10[9] * x + W_10[10]

    # Error de validación con k_folds
    # mean_MSE_lr = k_folds(X_train, y_train, 5, regression)
    print(X_quadratic.shape)
    # mean_MSE_linear = k_folds(X_linear, y_linear.reshape(-1, 2), 6, regression)
    # print(mean_MSE_linear)
    # mean_MSE_quadratic = k_folds(X_quadratic, y_test.reshape(-1, X_quadratic.shape[1]), 5, regression)
    # mean_MSE_cubic = k_folds(X_cubic, y_test.reshape(-1, X_cubic.shape[1]), 5, regression)
    # print('MSE_linear:  {}\nMSE_quadratic:    {}\nMSE_cubic: {}'.format(mean_MSE_linear, mean_MSE_quadratic
    #                                                                     , mean_MSE_cubic))

    # PLOTS
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('Sin(x) - Fitting curves')

    # original
    plt.plot(x, noisy_y, 'o')

    # linear
    plt.plot(x, y_linear, '-')

    # quadratic
    plt.plot(x, y_quadratic, '-')

    # cubic
    plt.plot(x, y_cubic, '-')

    # 10 power
    plt.plot(x, y_10, '-')

    plt.legend(['noisy signal', 'linear', 'quadratic', 'cubic', '10th power'])
    plt.show()


if __name__ == '__main__':
    dataset = Data('G:\My Drive\AI\CURSO\intro_ia\income.data.csv')

    X_train, X_test, y_train, y_test = dataset.split(0.8)
    # k=5
    # ajuste = LinearRegression()
    # mean_MSE_lr = k_folds(X_train,y_train,k,ajuste)
    #
    # X_train, X_test, y_train, y_test = dataset.split(0.8)
    # ajuste = LinearRegressionWithB()
    # mean_MSE_lr_b = k_folds(X_train, y_train, k, ajuste)
    #
    # X_train, X_test, y_train, y_test = dataset.split(0.8)
    # ajuste = GradientDescent()
    # mean_MSE_gd = k_folds(X_train.reshape(-1, 1), y_train.reshape(-1, 1), k, ajuste)
    # print('MSE_lr:  {}\nMSE_lr_b:    {}\nMSE_gd: {}'.format(mean_MSE_lr, mean_MSE_lr_b, mean_MSE_gd))


    # # Regresiones analíticas
    # linear_regression = LinearRegression()
    # linear_regression.fit(X_train, y_train)
    # lr_y_hat = linear_regression.predict(X_test)
    #
    # linear_regression_b = LinearRegressionWithB()
    # linear_regression_b.fit(X_train, y_train)
    # lrb_y_hat = linear_regression_b.predict(X_test)
    #
    # constant_model = ConstantModel()
    # constant_model.fit(X_train, y_train)
    # ct_y_hat = constant_model.predict(X_test)
    #
    # mse = MSE()
    # lr_mse = mse(y_test, lr_y_hat)
    # lrb_mse = mse(y_test, lrb_y_hat)
    # ct_mse = mse(y_test, ct_y_hat)
    #
    # x_plot = np.linspace(0, 10, 10)
    # lr_y_plot = linear_regression.model * x_plot
    # lrb_y_plot = linear_regression_b.model[0] * x_plot + linear_regression_b.model[1]
    #
    # # gradient descent
    # gradient_descent = GradientDescent()
    # print('\n\n\nGRADIENT DESCENT VS LINEAR REGRESSION')
    # lr_1 = 0.001
    # amt_epochs_1 = 1000
    # start = time.time()
    # gradient_descent.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1), lr=lr_1, amt_epochs=amt_epochs_1)
    # W_manual = gradient_descent.model
    # time_1 = time.time() - start
    # W_real = linear_regression.model
    # print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.format(W_manual.reshape(-1), W_real, time_1))
    #
    # print('\n\n\nGRADIENT DESCENT VS LINEAR REGRESSION WITH B')
    # X_expanded = np.vstack((X_train, np.ones(len(X_train)))).T
    # lr_2 = 0.001
    # amt_epochs_2 = 100000
    # start = time.time()
    # gradient_descent.fit(X_expanded, y_train.reshape(-1, 1), lr=lr_2, amt_epochs=amt_epochs_2)
    # W_manual = gradient_descent.model
    # time_2 = time.time() - start
    # W_real = linear_regression_b.model
    # print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.
    #       format(W_manual.reshape(-1), W_real, time_2))
    #
    # # stochastic gradient descent
    # stochastic_gradient_descent = StochasticGradientDescent()
    # print('\n\n\nSTOCHASTIC GRADIENT DESCENT VS LINEAR REGRESSION WITH B')
    # X_expanded = np.vstack((X_train, np.ones(len(X_train)))).T
    # lr_3 = 0.05
    # amt_epochs_3 = 1000
    # start = time.time()
    # stochastic_gradient_descent.fit(X_expanded, y_train.reshape(-1, 1), lr=lr_3, amt_epochs=amt_epochs_3)
    # W_manual = stochastic_gradient_descent.model
    # time_3 = time.time() - start
    # W_real = linear_regression_b.model
    # print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.
    #       format(W_manual.reshape(-1), W_real, time_3))
    #
    # # mini batch gradient descent
    # mini_batch_gradient_descent = MiniBatchGradientDescent()
    # print('\n\n\nMINI BATCH GRADIENT DESCENT VS LINEAR REGRESSION WITH B')
    # X_expanded = np.vstack((X_train, np.ones(len(X_train)))).T
    # lr_4 = 0.05
    # amt_epochs_4 = 10000
    # start = time.time()
    # mini_batch_gradient_descent.fit(X_expanded, y_train.reshape(-1, 1), lr=lr_4, amt_epochs=amt_epochs_4)
    # W_manual = mini_batch_gradient_descent.model
    # time_4 = time.time() - start
    # W_real = linear_regression_b.model
    # print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.
    #       format(W_manual.reshape(-1), W_real, time_4))
    #
    # # PLOTS
    # plt.figure()
    # x_plot = np.linspace(1, 4, 4)
    # legend = ['GD', 'GD(B)', 'S-GD(B)', 'MB-GD(B)']
    #
    # plt.subplot(1, 3, 1)
    # plt.gca().set_title('Learning Rate')
    # y_plot = [lr_1, lr_2, lr_3, lr_4]
    # plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
    #          x_plot[3], y_plot[3], 'o')
    # plt.legend(legend)
    # for x, y in zip(x_plot, y_plot):
    #     plt.text(x, y, str(y))
    #
    # plt.subplot(1, 3, 2)
    # plt.gca().set_title('Epochs')
    # y_plot = [amt_epochs_1, amt_epochs_2, amt_epochs_3, amt_epochs_4]
    # plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
    #          x_plot[3], y_plot[3], 'o')
    # plt.legend(legend)
    # for x, y in zip(x_plot, y_plot):
    #     plt.text(x, y, str(y))
    #
    # plt.subplot(1, 3, 3)
    # plt.gca().set_title('Time')
    # y_plot = [time_1, time_2, time_3, time_4]
    # plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
    #          x_plot[3], y_plot[3], 'o')
    # plt.legend(legend)
    # for x, y in zip(x_plot, y_plot):
    #     plt.text(x, y, str(y))
    #
    # plt.show()

    # sin fitting example
    sin_fitting_example()
