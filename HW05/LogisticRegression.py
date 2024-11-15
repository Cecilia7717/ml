
import math
import numpy as np
import random

from typing import Tuple

def add_ones(X: np.ndarray) -> np.ndarray:
    return np.hstack((np.ones((X.shape[0], 1)), X))

def logistic(X: np.ndarray, coef: np.ndarray) -> float:
    """Compute the logistic (sigmoid) function."""
    z = np.dot(X, coef)
    return 1 / (1 + np.exp(-z))

def costJ(y: np.ndarray, y_pred: np.ndarray) -> float:#, w, lmda):
    """Compute cost J"""
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def SGD_update(coef: np.ndarray, X: np.ndarray, y: int, alpha: float) -> np.ndarray:
    """Perform a SGD update based on one example."""
    # Ensure coef is of float type
    coef = coef.astype(float)
    
    prediction = logistic(X, coef)
    gradient = (prediction - y) * X
    a = alpha * gradient
    coef -= a  # This will now work without type issues
    return coef



def fit_SGD(X: np.ndarray, y: np.ndarray, alpha: float, eps: float =1e-4, tmax: int =1000000, shuffle_data: bool = False) -> np.ndarray:
    n,p = X.shape
    #eta_input = eta
    coef = np.zeros(p)                 # coefficients
    err_list  = np.zeros((tmax,1))           # errors per iteration
    X = add_ones(X)  # Add bias term
    coef = np.zeros(X.shape[1])
    prev_cost = float('inf')
    # SGD loop
    for t in range(tmax):
        # Update learning rate based on iteration
        alpha = 1 / (t + 200)

        # shuffle rows!!!!
        if shuffle_data:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
        
        # iterate through examples
        for i in range(X.shape[0]):
            coef = SGD_update(coef, X[i], y[i], alpha)
        
        # Check convergence
        y_pred = np.array([logistic(x, coef) for x in X])
        cost = costJ(y, y_pred)
        if abs(prev_cost - cost) < eps:
            break
        prev_cost = cost

    # print('number of iterations: %d' % (t+1))
    # print("coef:", coef)

    return coef



# testing
def prediction(X: np.ndarray,coef: np.ndarray) -> np.ndarray:
    """Make predictions for each example using the coefficents"""
    X = add_ones(X)  # Add bias term
    return np.array([logistic(x, coef) for x in X])

def threshold(y_pred: np.ndarray) -> np.ndarray:
    y_thresh = np.where(y_pred >= 0.5, 1, 0)
    return y_thresh

def accuracy(X: np.ndarray,coef: np.ndarray,y: np.ndarray) -> float:
    m = len(y) # num test data
    """Compute the accuracy by making predictions"""
    y_pred = prediction(X, coef)
    y_pred = threshold(y_pred)
    return np.mean(y_pred == y)