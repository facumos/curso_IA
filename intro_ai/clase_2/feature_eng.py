import numpy as np


def z_score(X):
    return (X-np.mean(X, axis=0))/np.std(X)


def PCA_by_hand(X, d):
    X_z = X - np.mean(X, axis=0)
    # # 2
    cov_x=np.matmul(np.transpose(X_z),X_z)/(X_z.shape[0]-1)
    # # 3 w: eigenvalues v:eigenvectors
    w, v = np.linalg.eig(cov_x)
    # # 4
    orden = w.argsort()[::-1]
    w = w[orden]
    v = v[:,orden]
    # # 5
    return np.matmul(X_z, v[:, :d])

n_components=3
x = np.array([ [0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0] ])
x_2 = PCA_by_hand(x,n_components)
print(x_2)

# Missing values
#  Ej 9 y 10 usar mascaras para saber donde hay Nan

