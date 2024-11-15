import unittest
from math import log

import numpy as np
import LogisticRegression

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
      self.X = np.array([[0.5, 1.2, 1.5, 0.7],
                        [1.1, 0.3, 0.9, 0.8],
                        [0.9, 0.6, 1.2, 1.0]])
      self.y = np.array([0, 1, 0])

    def test_add_one1(self):
        X_new = LogisticRegression.add_ones(self.X)
        self.assertEqual(X_new.shape, (3,5))

    def test_add_one2(self):
        X_new = LogisticRegression.add_ones(self.X)
        np.testing.assert_array_equal(X_new[:,0], np.array([1.0, 1.0, 1.0]))

    def test_add_one3(self):
        X_new = LogisticRegression.add_ones(self.X)
        np.testing.assert_array_equal(X_new[:,1:], self.X)

    def test_logistic1(self):
        output = LogisticRegression.logistic(self.X, np.zeros(4))
        np.testing.assert_array_equal(output, np.array([0.5, 0.5, 0.5]))

    def test_logistic2(self):
        output = LogisticRegression.logistic(np.zeros(4), np.random.random(4))
        np.testing.assert_array_equal(output, np.array([0.5, 0.5, 0.5]))

    def test_logistic3(self):
        output = LogisticRegression.logistic(np.ones(1), np.ones(1))
        self.assertAlmostEqual(output, 0.7310585786300049)

    def test_logistic4(self):
        output = LogisticRegression.logistic(self.X, np.ones(4))
        np.testing.assert_allclose(output, np.array([0.98015969, 0.95689275, 0.97587298]))

    def test_SGD_update1(self):
        coef = LogisticRegression.SGD_update(np.zeros(4), self.X[0], self.y[0], 1)
        np.testing.assert_array_equal(coef, np.array([-0.25, -0.6 , -0.75, -0.35]))

    def test_SGD_update2(self):
        coef = LogisticRegression.SGD_update(np.zeros(4), self.X[0], self.y[0], 10)
        np.testing.assert_array_equal(coef, np.array([-2.5, -6.,  -7.5, -3.5]))

    def test_SGD_update3(self):
        coef = LogisticRegression.SGD_update(np.zeros(4), self.X[1], self.y[1], 0.1)
        np.testing.assert_allclose(coef, np.array([0.055, 0.015, 0.045, 0.04 ]))

    def test_SGD_update4(self):
        coef = LogisticRegression.SGD_update(np.array([0,1,0,1]), self.X[1], self.y[1], 1)
        np.testing.assert_allclose(coef, np.array([0.27471388, 1.07492197, 0.2247659 , 1.19979192]), 0.001)

if __name__ == '__main__':
    unittest.main()
