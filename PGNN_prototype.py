"""
PGNN_prototype.py
Version: 1.0.0
Date: 2024/12/19
Author: Elia Arnese-Feffin elia249@mit.edu

# GNU General Public License version 3 (GPL-3.0) ------------------------------

PGNN_prototype.py
Copyright (C) 2024-2025 Elia Arnese-Feffin

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see https://www.gnu.org/licenses/gpl-3.0.html.

-------------------------------------------------------------------------------

To attribute credit to the author of the software, please refer to the
companion Journal Paper:
    E. Arnese-Feffin, N. Sagar, L. A. Briceno-Mena, B. Braun, I. Castillo\ap{3}, L. Bui, J. Xu, L. H. Chiang, and R. D. Braatz (2025):
        <TITLE>.
        <JOURNAL>, 00, 000-000.
        DOI: <DOI>.

"""

#%% Load packages

# Numerical Python
import numpy as np

#%% Articial Neural Networks (ANNs)

def SSE(Y, Y_pred):
    '''
    Sum of squared-error (SSE) function
    '''    
    return np.sum((Y - Y_pred)**2, axis = 0)

def MSE(Y, Y_pred):
    '''
    Mean squared-error (MSE) function
    '''
    return SSE(Y, Y_pred)/Y.shape[0]

def identity(x):
    '''
    Identity activation function
    '''
    return x

def tanh(x):
    '''
    Hyperbolic tangent activation function
    '''
    return np.tanh(x)

def logistic(x):
    '''
    Logistic activation function
    '''
    return (1 + np.exp(-x))**-1

def relu(x):
    '''
    Rectified Linear Unit (ReLU) activation function
    '''
    return np.maximum(0, x)

def identity_der(x):
    '''
    Derivative of the identity activation function
    '''
    return np.ones_like(x)

def logistic_der(x):
    '''
    Derivative of the hyperbolic tangent activation function
    '''
    f = logistic(x)
    return f*(1 - f)

def tanh_der(x):
    '''
    Derivative of the hyperbolic tangent activation function
    '''
    f = tanh(x)
    return 1 - f**2

def relu_der(x):
    '''
    Derivative of the rectified Linear Unit (ReLU) activation function
    '''
    return (x > 0).astype(x.dtype)

def layer(x, W, b, f):
    '''
    Computation layer of the neural network
    '''
    return f(b + W@x)

def ANN(x, W, b, f):
    '''
    Neural network function. W is a list containing the weights of each
    computational layer in the ANN, and b is a list of biases f each layer; f
    is a list of function handles to the activation functions of each layer.
    '''
    L = len(W)
    out = x
    for l in range(0, L):
        out = layer(out, W[l], b[l], f[l])
    return out

def ANN_detailed(x, W, b, f):
    '''
    Like ANN but returns every intermediate output of the ANN.
    '''
    L = len(W)
    layers = [None]*(L + 1)
    layers_act = [None]*(L + 1)
    layers[0] = x
    layers_act[0] = x
    for l in range(L):
        layers[l + 1] = b[l] + W[l]@layers_act[l]
        layers_act[l + 1] = f[l](layers[l + 1])
    return (layers, layers_act)

def ANN_params_to_vector(W, b, L, Ns, Np):
    '''
    Unfold lists of parameters into a single vector. Ns is a list of the numbers
    of neurons in each layer (including the input), while Np is the total number
    or parameters of the netowrk. L is the numebr of computational layers in the
    ANN (len(Ns) - 1).
    '''
    # Ravel parameters
    idx1 = 0
    idx2 = 0
    idx3 = 0
    p = np.empty(Np)
    for l in range(L):
        idx2 = idx1 + Ns[l + 1]
        idx3 = idx2 + Ns[l]*Ns[l + 1]
        p[idx1:idx2] = b[l].ravel()
        p[idx2:idx3] = W[l].ravel()
        idx1 = idx3
    return p

def ANN_params_to_matrices(p, L, Ns):
    '''
    Folds a vector of parameters intoto lsitslists of weights and biases.
    '''
    # Reconstruct parameter matrices
    idx1 = 0
    idx2 = 0
    idx3 = 0
    b = [None]*L
    W = [None]*L
    for l in range(L):
        idx2 = idx1 + Ns[l + 1]
        idx3 = idx2 + Ns[l]*Ns[l + 1]
        b[l] = p[idx1:idx2].reshape(Ns[l + 1], 1)
        W[l] = p[idx2:idx3].reshape(Ns[l + 1], Ns[l])
        idx1 = idx3
    return (W, b)

def ANN_jacobian_to_matrix(J_W, J_b, L, Ns, Np):
    '''
    Unfolds list of jacobian matrices of each parameter into a single matrix.
    '''
    # Number of observations in jacobian
    N = J_W[0].shape[2]
    # Ravel parameters
    idx1 = 0
    idx2 = 0
    idx3 = 0
    J = np.empty((Np, N))
    for l in range(L):
        idx2 = idx1 + Ns[l + 1]
        idx3 = idx2 + Ns[l]*Ns[l + 1]
        J[idx1:idx2, :] = J_b[l].reshape(-1, N)
        J[idx2:idx3, :] = J_W[l].reshape(-1, N)
        idx1 = idx3
    return J

def ANN_gradient(p, X, L, Ns, Np, f):
    '''
    Jacobian of the ANN function with respect fo each one of the observations
    passed as input: rows are parameters, columns are observations.
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    layers, layers_act = ANN_detailed(X, W, b, f)
    # Determine derivative functions
    f_der = [globals()[ff.__name__ + '_der'] for ff in f]
    # Compute errors
    E = [None]*(L + 1)
    E[L] = np.ones_like(layers_act[-1])
    # Get gradients
    delta = [None]*L
    J_b = [None]*L
    J_W = [None]*L
    for l in range(L):
        delta[L - l - 1] = E[L - l]*f_der[L - l - 1](layers[L - l]) # (Ns[L - l], N)
        J_b[L - l - 1] = delta[L - l - 1] # (Ns[L - l], N)
        J_W[L - l - 1] = np.empty((Ns[L - l], Ns[L - l - 1], X.shape[1])) # (Ns[L - l], Ns[L - l - 1], N)
        for n in range(Ns[L - l]):
            J_W[L - l - 1][n, :, :] = delta[L - l - 1][n, :]*layers_act[L - l - 1] # (1, Ns[L - l - 1], N)
        E[L - l - 1] = W[L - l - 1].T@delta[L - l - 1] # (Ns[L - l - 1], N)
        # Note that E[0] has no meaning, just needed for the loop
    # Ravel Jacobianmatrix
    J = ANN_jacobian_to_matrix(J_W, J_b, L, Ns, Np)
    return J

def ANN_objective(p, X, Y, L, Ns, Np, f):
    '''
    Objective function for ANN training based on MSE
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    Y_pred = ANN(X, W, b, f)
    # MSE
    O = np.sum((Y - Y_pred)**2)/(2*np.prod(np.shape(Y)))
    return O

def ANN_objective_gradient(p, X, Y, L, Ns, Np, f):
    '''
    Gradient of the objective function for ANN training based on MSE
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    layers, layers_act = ANN_detailed(X, W, b, f)
    # Compute errors
    E = [None]*(L + 1)
    E[L] = Y - layers_act[-1]
    # Determine derivative functions
    f_der = [globals()[ff.__name__ + '_der'] for ff in f]
    # Get gradients
    delta = [None]*L
    J_b = [None]*L
    J_W = [None]*L
    for l in range(L):
        delta[L - l - 1] = E[L - l]*f_der[L - l - 1](layers[L - l]) # (Ns[L - l], N)
        J_b[L - l - 1] = (-1/np.prod(np.shape(Y)))*np.sum(delta[L - l - 1], axis = 1, keepdims = True) # (Ns[L - l], 1)
        J_W[L - l - 1] = (-1/np.prod(np.shape(Y)))*delta[L - l - 1]@layers_act[L - l - 1].T # (Ns[L - l], Ns[L - l - 1])
        E[L - l - 1] = W[L - l - 1].T@delta[L - l - 1] # (Ns[L - l - 1], N)
        # Note that E[0] has no meaning, just needed for the loop
    # Ravel gradients
    J = ANN_params_to_vector(J_W, J_b, L, Ns, Np)
    return J

#%% Physics-Guided Neural Networks (PGNNs)

def ANN_gradient_L1(p, X, Y, L, Ns, Np, f):
    '''
    Gradient equality constraints for PGNN training
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    layers, layers_act = ANN_detailed(X, W, b, f)
    # Compute errors
    E = [None]*(L + 1)
    E[L] = np.sign(Y - layers_act[-1])
    # Determine derivative functions
    f_der = [globals()[ff.__name__ + '_der'] for ff in f]
    # Get gradients
    delta = [None]*L
    J_b = [None]*L
    J_W = [None]*L
    for l in range(L):
        delta[L - l - 1] = E[L - l]*f_der[L - l - 1](layers[L - l]) # (Ns[L - l], N)
        J_b[L - l - 1] = (-1/np.prod(np.shape(Y)))*np.sum(delta[L - l - 1], axis = 1, keepdims = True) # (Ns[L - l], 1)
        J_W[L - l - 1] = (-1/np.prod(np.shape(Y)))*delta[L - l - 1]@layers_act[L - l - 1].T # (Ns[L - l], Ns[L - l - 1])
        E[L - l - 1] = W[L - l - 1].T@delta[L - l - 1] # (Ns[L - l - 1], N)
        # Note that E[0] has no meaning, just needed for the loop
    # Ravel gradients
    J = ANN_params_to_vector(J_W, J_b, L, Ns, Np)
    return J

def ANN_gradient_lb(p, X, Y, L, Ns, Np, f):
    '''
    Gradient lower bound constraints for PGNN training
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    layers, layers_act = ANN_detailed(X, W, b, f)
    # Compute errors
    E = [None]*(L + 1)
    E[L] = relu_der(Y - layers_act[-1])
    # Determine derivative functions
    f_der = [globals()[ff.__name__ + '_der'] for ff in f]
    # Get gradients
    delta = [None]*L
    J_b = [None]*L
    J_W = [None]*L
    for l in range(L):
        delta[L - l - 1] = E[L - l]*f_der[L - l - 1](layers[L - l]) # (Ns[L - l], N)
        J_b[L - l - 1] = (-1/np.prod(np.shape(Y)))*np.sum(delta[L - l - 1], axis = 1, keepdims = True) # (Ns[L - l], 1)
        J_W[L - l - 1] = (-1/np.prod(np.shape(Y)))*delta[L - l - 1]@layers_act[L - l - 1].T # (Ns[L - l], Ns[L - l - 1])
        E[L - l - 1] = W[L - l - 1].T@delta[L - l - 1] # (Ns[L - l - 1], N)
        # Note that E[0] has no meaning, just needed for the loop
    # Ravel gradients
    J = ANN_params_to_vector(J_W, J_b, L, Ns, Np)
    return J

def ANN_gradient_ub(p, X, Y, L, Ns, Np, f):
    '''
    Gradient upper bound constraints for PGNN training
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    layers, layers_act = ANN_detailed(X, W, b, f)
    # Compute errors
    E = [None]*(L + 1)
    E[L] = relu_der(layers_act[-1] - Y)
    # Determine derivative functions
    f_der = [globals()[ff.__name__ + '_der'] for ff in f]
    # Get gradients
    delta = [None]*L
    J_b = [None]*L
    J_W = [None]*L
    for l in range(L):
        delta[L - l - 1] = E[L - l]*f_der[L - l - 1](layers[L - l]) # (Ns[L - l], N)
        J_b[L - l - 1] = (1/np.prod(np.shape(Y)))*np.sum(delta[L - l - 1], axis = 1, keepdims = True) # (Ns[L - l], 1)
        J_W[L - l - 1] = (1/np.prod(np.shape(Y)))*delta[L - l - 1]@layers_act[L - l - 1].T # (Ns[L - l], Ns[L - l - 1])
        E[L - l - 1] = W[L - l - 1].T@delta[L - l - 1] # (Ns[L - l - 1], N)
        # Note that E[0] has no meaning, just needed for the loop
    # Ravel gradients
    J = ANN_params_to_vector(J_W, J_b, L, Ns, Np)
    return J

def ANN_gradient_monotonicity(p, X, L, Ns, Np, f, diff_fun):
    '''
    Gradient monotonicity constraints for PGNN training
    '''
    # Jacobian of the ANN
    jac = ANN_gradient(p, X, L, Ns, Np, f)
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Run whole network
    Y_pred = ANN(X, W, b, f)
    # Apply differentiation function
    Y_pred_fd = diff_fun(Y_pred)
    # Compute gradient
    J = (diff_fun(jac)@relu_der(Y_pred_fd).T).flatten()/np.prod(np.shape(Y_pred))
    return J

def PGNN_objective(p, X, Y, L, Ns, Np, f, alpha = None, X_c = None, Y_lb = None, Y_ub = None, diff_fun = None, Y_eq = None):
    '''
    Objective function for PGNN training based on MSE
    '''
    # Reconstruct parameter matrices
    W, b = ANN_params_to_matrices(p, L, Ns)
    # Apply the ANN
    Y_pred = ANN(X, W, b, f)
    # MSE
    O = np.sum((Y - Y_pred)**2)/(2*np.prod(np.shape(Y)))
    
    # Check if collocation points are provided
    if X_c is not None:
        # Apply the ANN to the collocation points
        Y_c_pred = ANN(X_c, W, b, f)
        # Check if lower bound is stated
        if Y_lb is not None:
            # Lower bound for predictions
            lb = -(Y_c_pred - Y_lb)
            # Define arrays of lower inequality constraint
            g_lb = np.maximum(0, lb)
            # Update objective function
            O += alpha[0]*np.mean(g_lb, axis = 1)[0]
        # Check if upper bound is stated
        if Y_ub is not None:
            # Upper bound for predictions
            ub = Y_c_pred - Y_ub
            # Define arrays of upper inequality constraint
            g_ub = np.maximum(0, ub)
            # Update objective function
            O += alpha[1]*np.mean(g_ub, axis = 1)[0]
        # Check if monotonicity contraint is stated
        if diff_fun is not None:
            # Numerical derivative for monotonicity constraint
            dy_dX = diff_fun(Y_c_pred)
            # Define array of monotonicity constraints
            g_mc = np.maximum(0, dy_dX)
            # Update objective function
            O += alpha[2]*np.mean(g_mc, axis = 1)[0]
        # Check if equality constraints are stated
        if Y_eq is not None:
            # Deviation from reference values from FPM
            eq = Y_c_pred - Y_eq
            # Define array of equality constraints
            h_eq = np.abs(eq)
            # Update objective function
            O += np.dot(alpha[3:], np.mean(h_eq, axis = 1))
        
    return O

def PGNN_objective_gradient(p, X, Y, L, Ns, Np, f, alpha = None, X_c = None, Y_lb = None, Y_ub = None, diff_fun = None, Y_eq = None):
    '''
    Gradient of the objective function for PGNN training based on MSE
    '''
    # Gradient of ANN on loss function
    J = ANN_objective_gradient(p, X, Y, L, Ns, Np, f)
    
    # Check if collocation points are provided
    if X_c is not None:
        # Check if lower bound is stated
        if Y_lb is not None:
            # Gradient of lower bound
            J_lb = ANN_gradient_lb(p, X_c, np.full((1, X_c.shape[1]), Y_lb[0, 0]), L, Ns, Np, f)
            # Update gradient
            J += alpha[0]*J_lb
        # Check if upper bound is stated
        if Y_ub is not None:
            # Gradient of upper bound
            J_ub = ANN_gradient_ub(p, X_c, np.full((1, X_c.shape[1]), Y_ub[0, 0]), L, Ns, Np, f)
            # Update gradient
            J += alpha[1]*J_ub
        # Check if monotonicity contraint is stated
        if diff_fun is not None:
            # Gradient of monotonicity constraint
            J_mc = ANN_gradient_monotonicity(p, X_c, L, Ns, Np, f, diff_fun)
            # Update gradient
            J += alpha[2]*J_mc
        # Check if equality constraints are stated
        if Y_eq is not None:
            # Initialize gradient over collocation points
            J_eq = np.zeros_like(J)
            # Loop over reference values of outputs
            for i in range(Y_eq.shape[0]):
                # Gradient of ANN on collocation points
                J_eq += alpha[3 + i]*ANN_gradient_L1(p, X_c, Y_eq[i].reshape(1, -1), L, Ns, Np, f)
            # Update gradient
            J += J_eq
    
    return J