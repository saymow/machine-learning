# All implementations of the algorithms used to solve the notebooks problems 

import numpy as np


# Compute the S-shaped curve useful for logistic classification
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

# Computes polynomial regression (w*x + b)
def compute(w, b, x):
    return np.dot(x, w) + b

# Logistic loss
# Loss function for non linear functions
def compute_non_linear_cost(X, y, w, b, _):
    m, n = X.shape

    total_cost = 0
    
    for i in range(m):
        zwb_x = np.dot(X[i], w) + b
        fwb_x = sigmoid(zwb_x)
        loss = (-y[i] * np.log(fwb_x)) - ((1 - y[i]) * np.log(1 - fwb_x))
        total_cost += loss

    total_cost = total_cost / m
    
    return total_cost

# Squared loss
# Loss function for linear functions
def compute_linear_cost(X, y, w, b, _):
    total_cost = 0.0
    m = len(X)

    for i in range(m):
        total_cost += (compute(w, b, X[i]) - y[i]) ** 2
    total_cost = total_cost / (2 * m)
    
    return total_cost

# Computes regularized cost against cost_function
def compute_reg_cost(X, y, w, b, cost_function, lambda_ = 1):
    m, n = X.shape

    cost_without_reg = cost_function(X, y, w, b, lambda_)
    
    reg_cost = .0

    for j in range(n):
        reg_cost += w[j] ** 2

    reg_cost = lambda_ * reg_cost / (2 * m)
    
    return cost_without_reg + reg_cost 

# Gradient for regression
def compute_gradient_sigmoid(X, y, w, b, _):
    m, n = X.shape
    
    dj_db = 0.
    dj_dw = np.zeros(w.shape)

    for i in range(m):
        fwb_x = sigmoid(compute(w, b, X[i]))
        dj_db += fwb_x - y[i]

        for j in range(n):
            dj_dw[j] += (fwb_x - y[i]) * X[i][j]

    dj_db = dj_db / m
    dj_dw = dj_dw / m

    return dj_dw, dj_db

# Gradient for regression
def compute_gradient(X, y, w, b, _):
    m, n = X.shape
    
    dj_db = 0.
    dj_dw = np.zeros(w.shape)

    for i in range(m):
        fwb_x = compute(w, b, X[i])
        dj_db += fwb_x - y[i]

        for j in range(n):
            dj_dw[j] += (fwb_x - y[i]) * X[i][j]

    dj_db = dj_db / m
    dj_dw = dj_dw / m

    return dj_dw, dj_db

# Computes regularized gradient 
def compute_reg_gradient_sigmoid(X, y, w, b, lambda_ = 1):
    m, n = X.shape

    dj_dw, dj_db = compute_gradient_sigmoid(X, y, w, b, lambda_)

    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_dw, dj_db

# Computes gradient descent againts X and y
def compute_gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, lambda_, iterations):
    j_hist = []
    
    for idx in range(iterations):
        dj_dw, dj_db = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db


        if ((idx + 1) % 10 == 0 or idx == 0):
            cost = cost_function(X, y, w_in, b_in, lambda_)
            
            j_hist.append(cost)
            
            if ((idx + 1) % 1000 == 0 or idx == 0):
                print(f"Iteration {(idx + 1):6} | Cost {cost:8.2f}")

    return w_in, b_in, j_hist

# Computes mean normalization againts input data
def mean_normalization(x):
    means = np.mean(x, axis = 0)
    mins = np.min(x, axis = 0)
    maxs = np.max(x, axis = 0)

    normalized_x = (x - means) / (maxs - mins)
    
    return normalized_x

