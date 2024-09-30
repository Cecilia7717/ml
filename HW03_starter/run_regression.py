"""
Starter code authors: Yi-Chieh Wu, modified by Sara Mathieson, Adam Poliak
Authors:
Date:
Description:
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# import our model class
from PolynomialRegression import *

######################################################################
# main
######################################################################

def main() -> None :
    # toy data
    X = np.array([2]).reshape((1,1))     # shape (n,p) = (1,1)
    y = np.array([3]).reshape((1,))      # shape (n,) = (1,)
    coef = np.array([4,5]).reshape((2,)) # shape (p+1,) = (2,), 1 extra for bias

    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')


    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print('Visualizing data...')

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts b-e: main code for linear regression
    print('Investigating linear regression...')

    # model
    model = PolynomialRegression()

    # test part b -- soln: [[1 2]]
    print(model.generate_polynomial_features(X))

    # test part c -- soln: [14]
    model.coef_ = coef
    print(model.predict(X))

    # test part d, bullet 1 -- soln: 60.5
    print(model.cost(X, y))

    # test part d, bullets 2-3
    # for alpha = 0.01, soln: w = [2.441; -2.819], iterations = 616
    model.fit_SGD(train_data.X, train_data.y, 0.01)
    print('sgd solution: %s' % str(model.coef_))

    # test part e -- soln: w = [2.446; -2.816]
    model.fit(train_data.X, train_data.y)
    print('closed_form solution: %s' % str(model.coef_))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts f-h: main code for polynomial regression
    print('Investigating polynomial regression...')

    # toy data
    m = 2                                     # polynomial degree
    coefm = np.array([4,5,6]).reshape((3,))   # shape (3,), 1 bias + 3 coeffs

    # test part f -- soln: [[1 2 4]]
    model = PolynomialRegression(m)
    print(model.generate_polynomial_features(X))

    # test part g -- soln: 35.0
    model.coef_ = coefm
    print(model.rms_error(X, y))

    # non-test code (YOUR CODE HERE)

    # Check: RMSE for d=0 should be 0.747268364185172
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts i-j: main code for regularized regression
    print('Investigating regularized regression...')

    # test part i -- soln: [3 5.24e-10 8.29e-10]
    # note: your solution may be slightly different
    #       due to limitations in floating point representation
    #       you should get something close to [3 0 0]
    model = PolynomialRegression(m=2, reg_param=1e-5)
    model.fit(X, y)
    print(model.coef_)

    # non-test code (YOUR CODE HERE)

    ### ========== TODO : END ========== ###


    print("Done!")

if __name__ == "__main__" :
    main()
