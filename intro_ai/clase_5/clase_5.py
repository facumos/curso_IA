import time

import numpy as np
import matplotlib.pyplot as plt


from data import Data

from base_model import BaseModel, ConstantModel, LinearRegression, LinearRegressionWithB, Ridge
from base_model import GradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
from k_folds import k_folds_poly

from metric import Metric, MSE


# def k_folds(X_train, y_train, k=5, ajuste=LinearRegression()):
#     # l_regression = LinearRegression()
#     error = MSE()
#
#     chunk_size = int(len(X_train) / k)
#     mse_list = []
#     for i in range(0, len(X_train), chunk_size):
#         end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
#         new_X_valid = X_train[i: end]
#         new_y_valid = y_train[i: end]
#         new_X_train = np.concatenate([X_train[: i], X_train[end:]])
#         new_y_train = np.concatenate([y_train[: i], y_train[end:]])
#
#         ajuste.fit(new_X_train, new_y_train)
#         prediction = ajuste.predict(new_X_valid)
#         mse_list.append(error(new_y_valid, prediction))
#
#     mean_MSE = np.mean(mse_list)
#
#     return mean_MSE


def sin_fitting_example():
    # y = sin(x)

    # 1.Simular una función sin(x) con ruido
    amt_points = 3600
    x = np.linspace(0, 360, num=amt_points)
    y = np.sin(x * np.pi / 180.)
    noise = np.random.normal(0, .1, y.shape)  # Ruido para pruebas generales
    # noise = np.random.normal(0, 0.5, y.shape)   #Ruido para usar con Rigle
    noisy_y = y + noise

    # 2. Gráfico de los datos
    plt.plot(x, noisy_y, 'o')

    # Separo el dataset en train y test
    percentage = 0.8

    permuted_idxs = np.random.permutation(x.shape[0])
    # 2,1,3,4,6,7,8,5,9,0

    train_idxs = permuted_idxs[0:int(percentage * x.shape[0])]
    test_idxs = permuted_idxs[int(percentage * x.shape[0]): x.shape[0]]

    X_train = x[train_idxs]
    X_test = x[test_idxs]

    y_train = noisy_y[train_idxs]
    y_test = noisy_y[test_idxs]

    # plt.plot(X_train, y_train, 'o')

    # 5. Hacer fit para diferentes polinomios hasta 10
    # 6. Obtener mediante cross-validation para cada polinomio el error de validación (k-folds)

    # X_train = x
    # y_train = noisy_y

    regression = LinearRegression()

    mean_MSE_linear, mean_MSE_quadratic, mean_MSE_cubic, mean_MSE_4, mean_MSE_5, mean_MSE_6, mean_MSE_7, \
    mean_MSE_9,  mean_MSE_10 = k_folds_poly(X_train, y_train, 6, regression)

    # x_plot = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10])
    # mean_MSE=np.array([mean_MSE_linear, mean_MSE_quadratic, mean_MSE_cubic, mean_MSE_4, mean_MSE_5,
    #                    mean_MSE_6, mean_MSE_7, mean_MSE_9, mean_MSE_10])
    x_plot = np.array([3, 4, 5, 6, 7, 9, 10])
    mean_MSE=np.array([mean_MSE_cubic, mean_MSE_4, mean_MSE_5,
                       mean_MSE_6, mean_MSE_7, mean_MSE_9, mean_MSE_10])

    print('MSE_3:  {}\nMSE_5:    {}\nMSE_7: {}\nMSE_9: {}\nMSE_10: {}'.format(mean_MSE_cubic, mean_MSE_5, mean_MSE_7,
                                                                              mean_MSE_9, mean_MSE_10))

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('mean_MSE_errors - k_folds')
    plt.plot(x_plot, mean_MSE, '-')
    plt.show()

    # Seleccionar el modelo con complejidad correcta para el dataset
    # (usando el modelo que minimiza el validation error obtenido en 6)

    # El modelo correcto está entre polinomio de grado 3 y de grado 5

    # # Me genero una señal de test
    # amt_points = 360
    # x = np.linspace(0, 360, num=amt_points)
    # y = np.sin(x * np.pi / 180.)
    # noise = np.random.normal(0, 1, y.shape)
    # noisy_y = y + noise
    #
    # x_test = x
    # y_test = noisy_y

    # X5
    X_5 = np.vstack((np.power(X_train, 5), np.power(X_train, 4), np.power(X_train, 3),
                     np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regression.fit(X_5, y_train.reshape(-1, 1))
    W_5 = regression.model
    y_5 = W_5[0] * np.power(X_test, 5) + W_5[1] * np.power(X_test, 4) + \
          W_5[2] * np.power(X_test, 3) + W_5[3] * np.power(X_test, 2) + W_5[4] * X_test + W_5[5]

    error = MSE()

    print('MSE_5-test: {}'.format(error(y_test, y_5)))

    # Regularizar el modelo para mejorar la generalización del modelo (probar agregando mas ruido al sin(x))
    # Usando Ridge --> W=(X'X+lam*I)^(-1)dot(X.T).dot(y)
    regularization = Ridge()
    lam = 10000  # \lamda -- Probé con algunos lambda, los mejores resultados están alrededor de lam=10000
    X_5_rigle = np.vstack((np.power(X_train, 5), np.power(X_train, 4), np.power(X_train, 3),
                           np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regularization.fit(X_5_rigle, y_train.reshape(-1, 1), lam)
    W_5_rigle = regularization.model
    y_5_rigle = W_5_rigle[0] * np.power(X_test, 5) + W_5_rigle[1] * np.power(X_test, 4) + \
                W_5_rigle[2] * np.power(X_test, 3) + W_5_rigle[3] * np.power(X_test, 2) + W_5_rigle[4] * X_test +\
                W_5_rigle[5]

    print('MSE_5-test_rigle: {}'.format(error(y_test, y_5_rigle)))

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('ajuste con datos test')
    plt.plot(X_test, y_test, 'o')
    plt.plot(X_test, y_5, 'o')
    plt.plot(X_test, y_5_rigle, 'o')
    plt.legend(['data', 'Linear', 'Rigle'])
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
