"""
Starter code authors: Yi-Chieh Wu, modified by Sara Mathieson, Adam Poliak
Authors: Cecilia Chen
Date: 09/30/2024
Description: 
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Data :

    def __init__(self, X=None, y=None) -> None:
        """
        Data class.
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        """
        # n = number of examples, p = dimensionality
        self.X = X
        self.y = y

    def load(self, filename: str) -> None:
        """
        Load csv file into X array of features and y array of labels.
        filename (string)
        """

        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, 'data', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def plot(self, **kwargs) -> None:
        """Plot data."""
        if 'color' not in kwargs :
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename: str) -> Data:
    data = Data()
    data.load(filename)
    return data

def plot_data(X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    data = Data(X, y)
    data.plot(**kwargs)

class PolynomialRegression:

    def __init__(self, m: int = 1, reg_param: float = 0) -> None:
        """
        Ordinary least squares regression.
        coef_ (numpy array of shape (p+1,)) -- estimated coefficients for the
            linear regression problem (these are the w's from in class)
        m_ (integer) -- order for polynomial regression
        lambda_ (float) -- regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param

    def generate_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        params: X (numpy array of shape (n,p)) -- features
        returns: Phi (numpy array of shape (n,1+p*m) -- mapped features
        """

        n,p = X.shape

        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        
        one = np.ones((n, 1))
        #print("X before concatenation:")
        #print(X)

        #print("Shape of X:", X.shape)

        # Concatenate the arrays along axis 1 and assign it back to X
        X = np.concatenate((one, X), axis=1)

        #print("Array of ones:")
        #print(one)

        #print("X after concatenation:")
        #print(X)


        # part f: modify to create matrix for polynomial model
        Phi = X

        ### ========== TODO : END ========== ###

        return Phi

    def fit_SGD(self, X:np.ndarray, y:np.ndarray, eta: float = 0.01,
                eps: float = 1e-10, tmax:int = 1000000, verbose:bool = False) -> None:
        """
        Finds the coefficients of a polynomial that fits the data using least
        squares stochastic gradient descent.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
            alpha   -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        """
        if self.lambda_ != 0 :
            raise Exception("SGD with regularization not implemented")

        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(w)$')
            plt.ion()
            plt.show()
        X = self.generate_polynomial_features(X) # map features
        n,p = X.shape
        self.coef_ = np.zeros(p)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        #print("bbbb{}".format(X))
        # SGD loop
        for t in range(tmax):

            # iterate through examples
            for i in range(n) :
                ### ========== TODO : START ========== ###
                # part d: update self.coef_ using one step of SGD
                # hint: you can simultaneously update all w's using vector math

                # print(self.predict(X)[i])
                # print(y[i])
                # print(self.coef_.shape)
                # print(X[i])
                
                self.coef_ = self.coef_ - eta * (self.coef_[0]+self.coef_[1]*X[i] - y[i]) * X[i][1]
                pass

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = self.coef_[0] + self.coef_[1] * X # change this line
            # print(y)
            # print(np.power(y - y_pred[:, 1], 2))
            print(np.sum(np.power(y - y_pred[:, 1], 2)) / float(n))
            err_list[t] = np.sum(np.power(y - y_pred[:, 1], 2)) / float(n)
            #print(err_list)[t]
            if np.isinf(err_list[t]) or np.isnan(err_list[t]):
                break
            else:
                print(err_list[t])
            ### ========== TODO : END ========== ###

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) < eps :
                break

            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print('number of iterations: %d' % (t+1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Finds the coefficients of a polynomial that fits the data using the
        closed form solution.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        """

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution

        # part i: include L_2 regularization

        ### ========== TODO : END ========== ###


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for X.
        Parameters:
            X       -- numpy array of shape (n,p), features
        Returns:
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part c: predict y
        # h_w(x) = w_0 + w_1x
        y_pred = self.coef_[0] + self.coef_[1] * X # for simple linear regression case

        #h_w(x) = (w^T)X
        # y_pred = np.matmul(self.coef_.T, X)
        ### ========== TODO : END ========== ###

        return y_pred

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the objective function.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        Returns:
            cost    -- float, objective J(b)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(b)
        cost = 0
        n, p = X.shape
        diff = (self.predict(X) - y)
        diff_sq = np.matmul(diff.T, diff)
        
        for i in range (n):
            cost = cost + diff_sq[i]
        
        cost = cost/2
        ### ========== TODO : END ========== ###
        return cost


    def rms_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the root mean square error.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        Returns:
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part g: compute RMSE
        error = 0
        ### ========== TODO : END ========== ###
        return error


    def plot_regression(self, xmin: int =0, xmax: int =1, n: int =50, **kwargs) -> None:
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()
