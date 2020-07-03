from unittest import TestCase

import numpy as np

from clase_1.accuracy import Accuracy


class AccuracyTestCase(TestCase):

    def test_accuracy(self):
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        # truth = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        # prediction = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        expected = np.float64(0.4)
        result = Accuracy().__call__(truth, prediction)
        np.testing.assert_equal(expected, result)
