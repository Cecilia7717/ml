import unittest
from math import log

import numpy as np
from PolynomialRegression import PolynomialRegression, load_data

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
      self.X = np.array([2]).reshape((1,1))          # shape (n,d) = (1L,1L)
      self.y = np.array([3]).reshape((1,))           # shape (n,) = (1L,)
      self.coef = np.array([4,5]).reshape((2,))      # shape (d+1,) = (2L,), 1 extra for bias

      # load data
      self.train_data = load_data('regression_train.csv')
      self.test_data = load_data('regression_test.csv')

      self.model = PolynomialRegression()

    def test_generate_polynomial_features1(self):
        feats = self.model.generate_polynomial_features(self.X)
        self.assertEqual(feats[0][0], 1.0)
        self.assertEqual(feats[0][1], 2.0)
        train_feats = self.model.generate_polynomial_features(self.train_data.X)
        self.assertEqual(len(set(train_feats[:,0])), 1)

    def test_fit_SGD(self):
        self.model.fit_SGD(self.X, self.y, tmax=1)
        self.assertEqual(self.model.coef_[0], 0.03)
        self.assertEqual(self.model.coef_[1], 0.06)

        self.model.fit_SGD(self.X, self.y, tmax=2)
        self.assertAlmostEqual(self.model.coef_[0], 0.0585)
        self.assertAlmostEqual(self.model.coef_[1], 0.117)

        self.model.fit_SGD(self.X, self.y, tmax=3)
        self.assertEqual(self.model.coef_[0], 0.085575)
        self.assertEqual(self.model.coef_[1], 0.17115)

    def test_predict(self):
        self.model.fit_SGD(self.X, self.y)
        preds = self.model.predict(self.X)
        self.assertNotEqual(preds[0], 3)
        self.assertAlmostEqual(preds[0], 2.99997083)

    def test_predict2(self):
        self.model.fit_SGD(self.train_data.X, self.train_data.y)
        preds = np.round(self.model.predict(self.train_data.X))
        self.assertEqual(preds[0], 1)

    def test_cost2(self):
        self.model.coef_ = np.array([1,3])
        self.assertEqual(8.0, self.model.cost(self.X, self.y))

    def test_regularization(self):
        model = PolynomialRegression(reg_param=2)
        X = np.array([1,2,3]).reshape(3,1)
        y = np.array([.2,.3,.4])
        model.fit(X,y)
        self.assertAlmostEqual(model.coef_[0], 0.2)
        self.assertAlmostEqual(model.coef_[1], 0.05)

    def test_rmse(self):
        self.model.coef_ = np.array([1,3])
        self.assertEqual(self.model.rms_error(self.X, self.y), 4.0)

    def test_rmse(self):
        X = np.array([1,2,3]).reshape(3,1)
        y = np.array([1,4,13])
        self.model.coef_ = np.array([1,3])
        self.assertEqual(self.model.rms_error(X, y), 3.0)

if __name__ == '__main__':
    unittest.main()
