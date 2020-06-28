from unittest import TestCase

import numpy as np

from clase_1.metrics import obtain_var, calc_precision, calc_recall, calc_accuracy


class MetricsTestCase(TestCase):

    def test_obtain_var(self):
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        expected = np.array([3, 1, 3, 3])
        result = obtain_var(truth, prediction)
        np.testing.assert_equal(expected, result)

    def test_presicion(self):
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        expected = np.float64(0.5)
        var = obtain_var(truth, prediction)
        result = calc_precision(var)
        np.testing.assert_equal(expected, result)


    def test_recall(self):
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        expected = np.float64(0.5)
        var = obtain_var(truth, prediction)
        result = calc_recall(var)
        np.testing.assert_equal(expected, result)

    def test_accuracy(self):
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        expected = np.float64(0.4)
        var = obtain_var(truth, prediction)
        result = calc_accuracy(var)
        np.testing.assert_equal(expected, result)
