import numpy as np


class BaseModel(object):
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        W = Y.mean()
        self.model = W

    def predict(self, X):
        return np.ones(len(X)) * self.model


class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return self.model * X


class LinearRegressionWithB(BaseModel):

    def fit(self, X, y):
        X_expanded = np.vstack((X, np.ones(len(X)))).T
        W = np.linalg.inv(X_expanded.T.dot(X_expanded)).dot(X_expanded.T).dot(y)
        self.model = W

    def predict(self, X):
        X_expanded = np.vstack((X, np.ones(len(X)))).T
        return X_expanded.dot(self.model)


class Ridge(BaseModel):

    def fit(self, X, y, lam):
        if len(X.shape) == 1:
            W = X.T.dot(y) / (X.T.dot(X)+lam)
        else:
            W = np.linalg.inv(X.T.dot(X)+lam*np.identity(X.shape[1])).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return self.model * X


class GradientDescent(BaseModel):

    def fit(self, X, y, lr=0.01, amt_epochs=100):
        """
        shapes:
            X_t = nxm
            y_t = nx1
            W = mx1
        """
        n = X.shape[0]
        m = X.shape[1]

        # initialize random weights
        W = np.random.randn(m).reshape(m, 1)

        for i in range(amt_epochs):
            prediction = np.matmul(X, W)  # nx1
            error = y - prediction  # nx1

            grad_sum = np.sum(error * X, axis=0)
            grad_mul = -2 / n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

        self.model = W

    def predict(self, X):
        return self.model * X


class StochasticGradientDescent(BaseModel):

    def fit(self, X, y, lr=0.01, amt_epochs=100):
        """
            shapes:
                X_t = nxm
                y_t = nx1
                W = mx1
            """
        n = X.shape[0]
        m = X.shape[1]

        # initialize random weights
        W = np.random.randn(m).reshape(m, 1)

        for i in range(amt_epochs):
            idx = np.random.permutation(X.shape[0])
            X_train = X[idx]
            y_train = y[idx]

            for j in range(n):
                prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
                error = y_train[j] - prediction  # 1x1

                grad_sum = error * X_train[j]
                grad_mul = -2 / n * grad_sum  # 2x1
                gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1

                W = W - (lr * gradient)

        self.model = W

    def predict(self, X):
        return self.model * X


class MiniBatchGradientDescent(BaseModel):

    def fit(self, X, y, lr=0.01, amt_epochs=100):
        """
        shapes:
            X_t = nxm
            y_t = nx1
            W = mx1
        """
        b = 16
        n = X.shape[0]
        m = X.shape[1]

        # initialize random weights
        # W = np.random.randn(m).reshape(m, 1)
        W = np.array([[-3.10831876e-11], [-1.00425579e-06], [6.05261793e-04], [1.01313276e-03], [1.78439150e+01]])

        for j in range(amt_epochs):
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

            batch_size = int(len(X) / b)
            for i in range(0, len(X), batch_size):
                end = i + batch_size if i + batch_size <= len(X) else len(X)
                batch_X = X[i: end]
                batch_y = y[i: end]

                prediction = np.matmul(batch_X, W)  # nx1
                error = batch_y - prediction  # nx1

                grad_sum = np.sum(error * batch_X, axis=0)
                grad_mul = -2 / b * grad_sum  # 1xm
                gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

                W = W - (lr * gradient)

        self.model = W

    def predict(self, X):
        return self.model * X