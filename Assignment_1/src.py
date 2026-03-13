import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from cvxopt import matrix, solvers
np.random.seed(42)

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
        self.w = np.linalg.inv(X.T@X)@Y
        return self.w, np.linalg.cond(X.T@X)
    
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
        return self.w, np.linalg.cond(X.T @ X + lambdaa * np.eye(X.shape[1]))
    

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

        r = y - X @ w 

        for iteration in range(max_iter):
            w_old = w.copy()

            for j in range(d):
                rho = X[:, j].T @ (r + X[:, j] * w[j])

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
        non_zero = np.count_nonzero(w)/len(w)
        return self.w, non_zero


#Problem 3.5
class logistic_regression():

    def __init__(self, data):
        data = data[data[:,-1] != 0]
        self.X = data[:,:-1]
        self.y = data[:,-1]
        self.y[self.y == 1] = 0
        self.y[self.y == 2] = 1
        self.w = None
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def loss(self, W, X, y, laambda):
        """Calculate the value of the loss function"""
        N = X.shape[0]
        H = self.sigmoid(X@W)
        cross_entropy = -(1/N)*np.sum(y*np.log(H+1e-12) + (1-y)*np.log(1-H+1e-12))
        regularization = (laambda/(2*N))*np.linalg.norm(W)**2
        return cross_entropy + regularization

    
    def lipschitz_constant(self, laambda):
        """Compute the Lipschitz smoothness constant of the gradient."""
        X = self.X
        if X.ndim == 1:
            X = add_bias_1d(X)
        else:
            X = add_bias(X)
        N = X.shape[0]

        sigma_max_sq = np.linalg.norm(X, ord=2) ** 2
        L = sigma_max_sq / (4 * N) + laambda / N
        return L

    def logistic_regression(self, alpha, laambda, n_iters = 2000):
        """Vectorized Logistic Regression with l2 regularization"""
        X = self.X
        y = self.y

        if X.ndim == 1:
            X = add_bias_1d(X)
        else:
            X = add_bias(X)
        
        y = y.reshape(-1, 1)
        W = np.zeros((X.shape[1], 1))
        N = X.shape[0]
        loss_track = []

        for _ in range(n_iters):
            gradient = (1 / N) * (X.T @ (self.sigmoid(X@W) - y) + laambda*W)
            W = W - alpha * gradient
            loss = self.loss(W,X,y,laambda)
            loss_track.append(loss)

        self.w = W.reshape(W.shape[0],)

        plt.figure(figsize=(10,6))
        plt.plot(np.arange(n_iters), loss_track)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Loss vs iteration for learning rate {alpha}")
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.show()

        return self.w, loss_track


#Problem 3.6, 3.7 and 3.8
class gaussian_mixture():

    def __init__(self, data):
        self.X = data[:,:-1]
        self.X1 = data[:,:2]
    
    def marginal_log_likelihood(self, alpha, mu, sigma):
        """Calculates the marginal log likelihood"""
        K = alpha.shape[0]
        P = np.column_stack([
            alpha[k] * sp.stats.multivariate_normal.pdf(self.X, mean=mu[k], cov=sigma[k])
            for k in range(K)
        ])
        mixture = P.sum(axis=1)            
        mll = np.sum(np.log(mixture))
        return mll

    def expectation_maximization(self, K, iter, crash = False):
        """Updates the parameters. Runs the whole EM algorithm"""
        #initialization
        N = self.X.shape[0]
        alpha = (1/K)*np.ones(K)
        idx = np.random.randint(0,N, size=(K))
        mu = self.X[idx]
        data_var = np.mean(np.var(self.X, axis=0))
        f = np.random.rand(K,)*data_var
        sigma = np.array([s*np.eye(self.X.shape[1]) for s in f])

        intial_ll = self.marginal_log_likelihood(alpha, mu, sigma)
        ll_after = []

        for i in range(iter):
            #update the parameters
            R = np.column_stack([alpha[k] * sp.stats.multivariate_normal.pdf(self.X, mean=mu[k], cov=sigma[k]) for k in range(K)])
            R /= R.sum(axis=1, keepdims=True)   

            # Update Alpha
            alpha_modified = R.sum(axis=0) / N  

            # Update mu
            mu_modified = np.zeros_like(mu)
            for k in range(K):
                mu_modified[k] = R[:, k] @ self.X / R[:, k].sum()

            #Update sigma
            sigma_modified = np.zeros_like(sigma)
            for k in range(K):
                diff = self.X - mu[k]                                  
                sigma_modified[k] = (R[:, k].reshape(-1,1) * diff).T @ diff / R[:, k].sum()
            
            if crash:
                crash_idx = np.random.randint(0,K)
                sigma_modified[crash_idx] = np.zeros((self.X.shape[1],self.X.shape[1]))

            alpha = alpha_modified
            mu = mu_modified
            sigma = sigma_modified
            ll_after.append(self.marginal_log_likelihood(alpha, mu, sigma))

        return intial_ll, ll_after, alpha, mu, sigma
    
    def decision_boundary(self, K=3, n_iters=10):
        """Makes the contour of optimal decision boundary"""
        X = self.X1                        
        N, D = X.shape

        # Run EM directly on 2D data (self.X1, not self.X)
        alpha = (1/K)*np.ones(K)
        idx = np.random.randint(0, N, size=K)
        mu = X[idx].copy()
        data_var = np.mean(np.var(X, axis=0))
        f = np.random.rand(K,) * data_var
        sigma = np.array([s*np.eye(D) for s in f])
        for _ in range(n_iters):
            R = np.column_stack([
                alpha[k] * sp.stats.multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
                for k in range(K)
            ])
            R /= R.sum(axis=1, keepdims=True)
            alpha = R.sum(axis=0) / N
            mu_new = np.zeros_like(mu)
            for k in range(K):
                mu_new[k] = R[:, k] @ X / R[:, k].sum()
            sigma_new = np.zeros((K, D, D))
            for k in range(K):
                diff = X - mu[k]
                sigma_new[k] = (R[:, k].reshape(-1,1) * diff).T @ diff / R[:, k].sum()
            mu, sigma = mu_new, sigma_new

        # Dense grid over 2D space
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Bayes optimal: argmax_k P(k|x) over grid
        P_grid = np.column_stack([
            alpha[k] * sp.stats.multivariate_normal.pdf(grid, mean=mu[k], cov=sigma[k])
            for k in range(K)
        ])
        labels = P_grid.argmax(axis=1).reshape(xx.shape)

        colors = plt.cm.tab10(np.linspace(0, 0.3, K))

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, labels, alpha=0.25, cmap='tab10')
        plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.2, c='gray')

        # Contour lines of each learned Gaussian covariance
        for k in range(K):
            Z_k = sp.stats.multivariate_normal.pdf(
                grid, mean=mu[k], cov=sigma[k]).reshape(xx.shape)
            plt.contour(xx, yy, Z_k, levels=4, colors=[colors[k]], linewidths=1.5)
            plt.scatter(*mu[k], marker='x', s=150, linewidths=2,
                        color=colors[k], zorder=5, label=f'Mean {k}')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'GMM Decision Boundaries (K={K}) with Gaussian Contours')
        plt.legend()
        plt.show()

        return None


#Problem 3.9,3.10,3.11
class soft_margin_classifier():

    def __init__(self, data):
        data = data[data[:,-1] != 0]
        self.X = data[:,:-1]
        self.y = data[:,-1]
        self.y[self.y == 1] = -1
        self.y[self.y == 2] = 1
        self.Q = np.outer(self.y, self.y) * (self.X @ self.X.T)   
        self.w = None
        data_rbf = data[:1000]
        self.X1 = data_rbf[:,:-1]
        self.y1 = data_rbf[:,-1]
        self.y1[self.y1 == 1] = -1
        self.y1[self.y1 == 2] = 1

    def optimal_mu(self, C):
        N = len(self.y)
        P = matrix(self.Q.astype(float))
        q = matrix(-np.ones(N, dtype=float))
        G = matrix(np.vstack([-np.eye(N), np.eye(N)]).astype(float))  
        h = matrix(np.hstack([np.zeros(N), np.ones(N) * C]).astype(float))
        A = matrix(self.y.astype(float), (1, N), 'd')                
        b = matrix([0.0])

        solvers.options['show_progress'] = True
        result = solvers.qp(P, q, G, h, A, b)

        return np.array(result['x']).flatten()
    
    def rbf_kernel(self, sigma, X, eval = False):
        square_norms = np.sum(X**2, axis=1)
        dist_sq = square_norms[:,None] + square_norms[None,:] - 2*X@X.T
        K = np.exp(-(dist_sq/sigma**2)) 
        if eval:
            evals = np.linalg.eigvals(K)
            return K, evals
        else:
            return K
        
    def hyperparameter_topography(self, C, sigma, train_idx, val_idx):
        X = self.X
        y = self.y
        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]
        K = self.rbf_kernel(sigma, X_train)
        Q = np.outer(y_train, y_train) * K  
    
        N = len(y_train)
        P = matrix(Q.astype(float))
        q = matrix(-np.ones(N, dtype=float))
        G = matrix(np.vstack([-np.eye(N), np.eye(N)]).astype(float))  
        h = matrix(np.hstack([np.zeros(N), np.ones(N) * C]).astype(float))
        A = matrix(y_train.astype(float), (1, N), 'd')                
        b = matrix([0.0])

        print(f"Iteration for C = {C}, sigma = {sigma}")
        solvers.options['show_progress'] = True
        result = solvers.qp(P, q, G, h, A, b)

        mu = np.array(result['x']).flatten()
        support_vectors = mu >1e-5
        no_support = np.sum(support_vectors == True)
        mu_support = mu[support_vectors]
        y_support = y_train[support_vectors]
        X_support = X_train[support_vectors]
        IP_val = np.vstack([X_support, X_val])
        K_val = self.rbf_kernel(sigma, IP_val)
        predicted = np.zeros_like(y_val)

        b = np.mean(y_support - (mu_support*y_support)@K_val[:no_support,:no_support])

        decision = (mu_support * y_support) @ K_val[:no_support, no_support:] + b
        predicted = np.sign(decision)

        K_train = K_val[:no_support, :no_support]
        decision_train = (mu_support*y_support) @ K_train + b
        train_pred = np.sign(decision_train)

        training_accuracy = np.sum(train_pred == y_support)/len(y_support)
        validation_accuracy = np.sum(predicted == y_val)/len(y_val)

        return training_accuracy, validation_accuracy
        
#Problem 4.12
class multi_class_logistic_regression():

    def __init__(self, data):
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def loss(self, W, X, y, laambda):
        """Calculate the value of the loss function"""
        N = X.shape[0]
        H = self.sigmoid(X@W)
        cross_entropy = -(1/N)*np.sum(y*np.log(H+1e-12) + (1-y)*np.log(1-H+1e-12))
        regularization = (laambda/(2*N))*np.linalg.norm(W)**2
        return cross_entropy + regularization

    def standard_scaler(self, X):
        mu = np.mean(X, axis= 0)
        sigma = np.std(X, axis=0)
        return ((X-mu)/sigma)

    def logistic_regression(self, alpha, laambda, n_iters = 2000, standard_scaling = False):
        """Vectorized Logistic Regression with l2 regularization"""
        X = self.X
        y = self.y

        if standard_scaling:
            X = self.standard_scaler(X)

        if X.ndim == 1:
            X = add_bias_1d(X)
        else:
            X = add_bias(X)
        
        y = y.reshape(-1, 1)
        W = np.zeros((X.shape[1], 1))
        N = X.shape[0]
        loss_track = []

        for n in range(n_iters):
            gradient = (1 / N) * (X.T @ (self.sigmoid(X@W) - y) + laambda*W)
            W = W - alpha * gradient
            loss = self.loss(W,X,y,laambda)
            loss_track.append(loss)

            if n>0:
                if np.abs((loss_track[-2] - loss_track[-1])) < 1e-6:
                    print(f"The regression converged in {n+1} iterations")
                    break
        return n+1, loss_track

#Problem 4.13,4.14
class k_nearest_neighbour():

    def __init__(self, data):
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def standard_scaler(self, X):
        mu = np.mean(X, axis= 0)
        sigma = np.std(X, axis=0)
        return mu, sigma
    
    def naive_knn(self, K, train_idx, test_idx, standard_scaler = False):
        """Uses loop for KNN"""
        X_test = self.X[test_idx]
        X_train = self.X[train_idx]
        y_test = self.y[test_idx]
        y_train = self.y[train_idx]

        if standard_scaler:
            mu, sigma = self.standard_scaler(X_train)
            X_train = (X_train - mu)/sigma
            X_test = (X_test-mu)/sigma
        
        predicted = np.zeros_like(y_test)
        for i,X in enumerate(X_test):
            distance_track = np.zeros_like(y_train)
            for j,X_t in enumerate(X_train):
                distance = np.linalg.norm(X - X_t)
                distance_track[j] = distance
            idx_knn = np.argpartition(distance_track, K)[:K]
            y_predicted = sp.stats.mode(y_train[idx_knn], keepdims=False).mode
            predicted[i] = y_predicted
        
        return predicted, y_test

    def vectorized_knn(self, K, train_idx, test_idx, standard_scaler = False):
        """Uses vectorized KNN code"""
        X_test = self.X[test_idx]
        X_train = self.X[train_idx]
        y_test = self.y[test_idx]
        y_train = self.y[train_idx]

        if standard_scaler:
            mu, sigma = self.standard_scaler(X_train)
            X_train = (X_train - mu)/sigma
            X_test = (X_test-mu)/sigma
        
        train_sq = np.sum(X_train**2, axis=1)
        test_sq = np.sum(X_test**2, axis=1)

        distance_matrix = train_sq[:, None] + test_sq[None, :] - 2 * X_train @ X_test.T
        idx_knn = np.argpartition(distance_matrix, K, axis=0)[:K]
        neighbors = y_train[idx_knn]
        predicted = sp.stats.mode(neighbors, axis=0, keepdims=False).mode

        return predicted, y_test

    

if __name__ == "__main__":
    data = load_data('dataset_3.csv')
    # X = data[:500,0]
    # y = data[:500,1]
    # y_binary = np.where(y>=np.median(y),1,-1)

    # model = hard_margin_svm(X, y_binary)

    # mu = model.optimal_mu_scipy()

    # print(mu)

    # model = logistic_regression(data)
    # L = model.lipschitz_constant(laambda=0.01)
    # print(f"Lipschitz constant L = {L:.4f}")
    # print(f"Max safe learning rate alpha < {1/L:.6f}")

    # # converges
    # model.logistic_regression(alpha=1/L, laambda=0.01)

    # # violates Lipschitz condition → diverging loss curve
    # model.logistic_regression(alpha=10/L, laambda=0.01)



    # initial_ll, modified_ll = model.expectation_maximization(4,10)
    # print(initial_ll, modified_ll)
    
    # model.decision_boundary()



    # model = ordinary_least_squares()

    # weights_normal = model.normal_equation_fit(data[:,0],data[:,1])

    # weights_gradient = model.gradient_descent_fit(data[:,0],data[:,1])
    # print(np.linalg.norm(weights_gradient-weights_normal))

    # print(weights_normal)
    # print(weights_gradient)

    model = soft_margin_classifier(data)
    mu_optimal = model.optimal_mu(1)
    print(mu_optimal[mu_optimal == 1])