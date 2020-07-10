from unittest import TestCase


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clase_2.feature_eng import z_score, PCA_by_hand


class FeatureEngTest(TestCase):

    def z_score_test(self):
        n_muestras = 100
        m_columnas = 10
        X = np.random.uniform(0, 1, size=(n_muestras, m_columnas))
        expected = 0
        result = np.mean(z_score(X))
        np.testing.assert_equal(expected, result)

    def PCA_by_hand_test(self):
        n_components=3
        x = np.array([[0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0]])
        pca = PCA(n_components)
        x_std = StandardScaler(with_std=False).fit_transform(x)
        expected = pca.fit_transform(x_std)
        result = PCA_by_hand(x)
        np.testing.assert_equal(expected, result)
