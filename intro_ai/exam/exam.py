import numpy as np
import matplotlib.pyplot as plt


from data import Data
from base_model import BaseModel, LinearRegression, MiniBatchGradientDescent, Ridge
from k_folds import k_folds_poly
from metric import Metric, MSE


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    # W = np.random.randn(m).reshape(m, 1)
    W = np.array([[-3.10831876e-11], [-1.00425579e-06], [6.05261793e-04], [1.01313276e-03], [1.78439150e+01]])
    chunk_size = 5
    part = int(X_train.shape[0]/chunk_size)

    new_X_valid = X_train[0: part]
    new_y_valid = y_train[0: part]
    new_X_train = np.concatenate([X_train[: 0], X_train[part:]])
    new_y_train = np.concatenate([y_train[: 0], y_train[part:]])

    # error_acum = 0

    for j in range(amt_epochs):
        idx = np.random.permutation(new_X_train.shape[0])
        new_X_train = new_X_train[idx]
        new_y_train = new_y_train[idx]
        batch_size = int(len(new_X_train) / b)
        for i in range(0, len(new_X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(new_X_train) else len(new_X_train)
            batch_X = new_X_train[i: end]
            batch_y = new_y_train[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            error_mb = batch_y - prediction  # nx1
            # error_acum(error_mb)

            grad_sum = np.sum(error_mb * batch_X, axis=0)  # mx1
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

    # error_acum = error_acum/b
    error_val = new_y_valid - np.matmul(new_X_valid, W)

    return W


if __name__ == '__main__':

# Preprocesamiento del dataset
    dataset = Data('G:\My Drive\AI\CURSO\intro_ia\clase_8_dataset.csv')

    X_train, X_test, y_train, y_test = dataset.split(0.8)

    plt.figure()
    plt.gca().set_title('Train Dataset')
    plt.plot(X_train, y_train, 'o')
    plt.show()


# 3. Regresión polinómica para hacer fit

    # a. Usar el modelo Regresión lineal con b

    regression = LinearRegression()  # Invoco la clase Linear Regression, que hereda de Base Model métodos

    # Utilizar K-Folds para entrenar

    partition = 5  # numero por el cual voy a partir el train dataset en partes iguales
    mean_MSE_linear, mean_MSE_quadratic, mean_MSE_cubic, mean_MSE_4 = k_folds_poly(X_train, y_train, partition, regression)

    # Selecciono el mejor modelo a partir del mínimo error cuadrático medio que entrega K-Folds

    print('MSE_1:  {}\nMSE_2:  {}\nMSE_3:  {}\nMSE_4:    {}'
          .format(mean_MSE_linear, mean_MSE_quadratic,mean_MSE_cubic, mean_MSE_4))

    x_plot = np.array([1, 2, 3, 4])
    mean_MSE = np.array([mean_MSE_linear, mean_MSE_quadratic, mean_MSE_cubic, mean_MSE_4])

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('Mean MSE errors - k_folds')
    plt.plot(x_plot, mean_MSE, '-')
    plt.show()

    # A partir de ver los resultados de calcular, comparando contra los datos de validación y tomando el promedio de los
    # errores cuadráticos medios, se decide que el polinomio que mejor ajusta es el de grado 4.

    # Ahora entreno el polinomio de grado 4 seleccionado y lo comparo con el test dataset para evaluar la predicción

    # X4
    X_4 = np.vstack((np.power(X_train, 4), np.power(X_train, 3),
        np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regression.fit(X_4, y_train.reshape(-1, 1))
    W_4 = regression.model
    y_4 = W_4[0] * np.power(X_test, 4) + W_4[1] * np.power(X_test, 3)+ \
        W_4[2] * np.power(X_test, 2) + W_4[3] * X_test+ W_4[4]

    error = MSE()

    print('MSE_4-test: {}'.format(error(y_test, y_4)))

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('Ajuste del test')
    plt.plot(X_test, y_test, 'o', X_test, y_4, 'o')
    plt.legend(['test dataset', 'ajuste elegido n=4'])
    plt.show()

# 4. Regresión polinómica para hacer fit

    # Vuelvo a utilizar X4 como dataset de entrada
    lr = 0.00000000000000000001  # Creo que le pegué al parámetro
    amt_epochs = 10000

    gradient = MiniBatchGradientDescent()
    # gradient.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1), lr, amt_epochs)
    gradient.fit(X_4, y_train.reshape(-1, 1), lr, amt_epochs)
    W_MB = gradient.model
    # W_MB_2 = mini_batch_gradient_descent(X_4, y_train.reshape(-1, 1), lr, amt_epochs)
    y_MB = W_MB[0] * np.power(X_test, 4) + W_MB[1] * np.power(X_test, 3)+ \
        W_MB[2] * np.power(X_test, 2) + W_MB[3] * X_test+ W_MB[4]
    error = MSE()

    print('MSE_MB-test: {}'.format(error(y_test, y_MB)))

# 5. Agrego Ridge
    ridge = Ridge()
    lam = 100
    X_MB = W_MB[0] * np.power(X_train, 4) + W_MB[1] * np.power(X_train, 3)+ \
        W_MB[2] * np.power(X_train, 2) + W_MB[3] * X_train+ W_MB[4]
    ridge.fit(X_MB, y_train, lam)
    W_R = ridge.model
    y_R = W_R[0] * np.power(X_test, 4) + W_R[1] * np.power(X_test, 3)+ \
        W_R[2] * np.power(X_test, 2) + W_R[3] * X_test+ W_R[4]
    error = MSE()

    print('MSE_R-test: {}'.format(error(y_test, y_R)))














