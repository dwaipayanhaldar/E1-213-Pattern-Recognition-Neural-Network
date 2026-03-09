import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from cvxopt import matrix, solvers
"""
Basic Utility functions
"""
def load_data(file_name):
    """Loads the dataset from a CSV file"""
    DATA_DIR = '/Users/dwaipayanhaldar/Downloads/Notes and Books/IISc Coding Assignments and Project/Data/PRNN_2026_A1_data'
    data = np.loadtxt(os.path.join(DATA_DIR, file_name), delimiter=',', skiprows=1)
    return data

def add_bias_1d(X):
    """For a 1d feature array, it adds another column of all ones, which is the bias term"""
    ones = np.ones_like(X)
    X = X.reshape(X.shape[0],1)
    ones = ones.reshape(ones.shape[0],1)
    X = np.hstack((X,ones))
    return X

def add_bias(X):
    """For not 1d feature array, it adds another column of all ones, which is the bias term"""
    ones = np.ones_like(X[:,0])
    ones = ones.reshape(ones.shape[0],1)
    X = np.hstack((X,ones))
    return X

"""
Main Algorithmic Functions
"""

#Problem 1.1
class ordinary_least_squares():
    def __init__(self):
        self.w = None

    def normal_equation_fit(self, X, y):
        """Solve the OLS problem using the normal equations"""
        if X.ndim == 1:
            X = add_bias_1d(X)
        else: 
            X = add_bias(X)
        Y = X.T @ y
        self.w = np.linalg.inv(X.T @ X) @ Y
        return self.w
    
    def gradient_descent_fit(self, X, y, alpha=0.01, n_iters=2000):
        """Solves the OLS problem using gradient descent"""
        if X.ndim == 1:
            X = add_bias_1d(X)
        else: 
            X = add_bias(X)
        y = y.reshape(-1, 1)
        W = np.zeros((X.shape[1], 1))
        N = X.shape[0]

        for _ in range(n_iters):
            gradient = (2 / N) * X.T @ (X @ W - y)
            W = W - alpha * gradient

        self.w = W.reshape(W.shape[0],)
        return self.w


#Problem 1.2
class hard_margin_svm():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.v = X * y   # vector of length N

    def opt_prob(self, mu):
        s = np.dot(self.v, mu)
        return 0.5 * s**2 - np.sum(mu)

    def opt_prob_jac(self, mu):
        s = np.dot(self.v, mu)
        return self.v * s - np.ones_like(mu)

    def optimal_mu(self):
        N = len(self.y)
        v = self.v.reshape(-1, 1)

        P = matrix(np.outer(v, v).astype(float))
        q = matrix(-np.ones(N, dtype=float))
        G = matrix(-np.eye(N).astype(float))   
        h = matrix(np.zeros(N, dtype=float))
        A = matrix(self.y.astype(float), (1, N), 'd')   
        b = matrix([0.0])

        solvers.options['show_progress'] = True
        result = solvers.qp(P, q, G, h, A, b)

        return np.array(result['x']).flatten()
    
    def optimal_mu_scipy(self):

        def eq_constraint(mu):
            return np.dot(mu, self.y)

        cons = {'type':'eq', 'fun': eq_constraint}
        bounds = [(0, None) for _ in range(len(self.y))]

        mu0 = np.ones_like(self.y) * 1e-3

        result = sp.optimize.minimize(
            self.opt_prob,
            mu0,
            jac=self.opt_prob_jac,
            method='SLSQP',
            constraints=cons,
            bounds=bounds,
            tol=1e-6
        )

        return result


#Problem 2.3 and 2.4
class regression():

    def __init__(self, data):
        self.X = data[:,:-1]
        self.y = data[:,-1]
        self.w = None

    def ordinary_least_squares(self):
        """Solve the OLS problem using the normal equations"""
        X = self.X
        y = self.y
        if X.ndim == 1:
            X = add_bias_1d(X)
        else: 
            X = add_bias(X)
        Y = X.T @ y
        self.w = np.linalg.solve(X.T@X, Y)
        return self.w
    
    def l2_regression(self, lambdaa):
        """Solve the L2 regression problem from normal equation"""
        X = self.X
        y = self.y
        if X.ndim == 1:
            X = add_bias_1d(X)
        else: 
            X = add_bias(X)
        Y = X.T @ y
        self.w = np.linalg.inv(X.T @ X + lambdaa * np.eye(X.shape[1])) @ Y
        return self.w
    

    def l1_regression(self, lambdaa, max_iter=1000, tol=1e-6):
        """Solve the L1 regression problem"""
        X = self.X
        y = self.y

        if X.ndim == 1:
            X = add_bias_1d(X)
        else:
            X = add_bias(X)

        n, d = X.shape
        w = np.zeros(d)
        z = np.sum(X**2, axis=0)

        for iteration in range(max_iter):
            w_old = w.copy()

            for j in range(d):
                r = y - X @ w + X[:, j] * w[j]

                rho = X[:, j].T @ r
                if rho > lambdaa:
                    w[j] = (rho - lambdaa) / z[j]
                elif rho < -lambdaa:
                    w[j] = (rho + lambdaa) / z[j]
                else:
                    w[j] = 0.0

            if np.linalg.norm(w - w_old) < tol:
                print(f"Converged in {iteration+1} iterations")
                break

        self.w = w
        return w



if __name__ == "__main__":
    data = load_data('dataset_2.csv')
    # X = data[:500,0]
    # y = data[:500,1]
    # y_binary = np.where(y>=np.median(y),1,-1)

    # model = hard_margin_svm(X, y_binary)

    # mu = model.optimal_mu_scipy()

    # print(mu)

    model = regression(data)

    print("The weights are:", model.ordinary_least_squares())



    # model = ordinary_least_squares()

    # weights_normal = model.normal_equation_fit(data[:,0],data[:,1])

    # weights_gradient = model.gradient_descent_fit(data[:,0],data[:,1])
    # print(np.linalg.norm(weights_gradient-weights_normal))

    # print(weights_normal)
    # print(weights_gradient)

