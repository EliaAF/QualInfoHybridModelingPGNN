"""
pH_case_study.py
Version: 1.0.0
Date: 2024/12/19
Author: Elia Arnese-Feffin elia249@mit.edu

# GNU General Public License version 3 (GPL-3.0) ------------------------------

pH_case_study.py
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
# Tabular data
import pandas as pd

# Data pre-processing
from sklearn.preprocessing import StandardScaler

# Optimization routine
from scipy.optimize import minimize

# Finite difference approximation of derivatives
import findiff as fd

# Plots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ANNs and PGNNs
from PGNN_prototype import MSE
from PGNN_prototype import tanh, identity#, tanh_der, identity_der
from PGNN_prototype import ANN, ANN_objective, ANN_objective_gradient
from PGNN_prototype import ANN_params_to_matrices
from PGNN_prototype import PGNN_objective, PGNN_objective_gradient

#%% Settings for plots

# Set resolution of plots
plt.rcParams['figure.dpi'] = 330

# Line thickness
lw = 1
# Marker size
ms = 2
# Marker edge line width
melw = ms*0.1
# Marker edge line colour
melc = [0.9, 0.9, 0.9]
# List of colours
colours = list(mcolors.TABLEAU_COLORS.values())
colours = [
    [0.5000, 0.5000, 0.5000],
    [0.1216, 0.4667, 0.7059],
    [1.0000, 0.4980, 0.0549],
    [0.1725, 0.6275, 0.1725],
    [0.8392, 0.1529, 0.1569],
    [0.5804, 0.4039, 0.7412],
    [0.5490, 0.3373, 0.2941],
    [0.8902, 0.4667, 0.7608],
    [0.4980, 0.4980, 0.4980],
    [0.7373, 0.7412, 0.1333],
    [0.0902, 0.7451, 0.8118]
]

# Settings for plots
small_fs = (3.15, 2.3625)
large_fs = (6.30, 4.725)
large_fontsize = 10
fontsize = 8
small_fontsize = 6

# Dictionary of options for title
opts_tit = {
    'fontsize' : large_fontsize
}
# Dictionary of options for labels
opts_labs = {
    'fontsize' : fontsize
}
# Dictionary of options for ticks
opts_ticks = {
    'direction' : 'in',
    'bottom' : True,
    'top' : True,
    'left' : True,
    'right' : True,
    'labelsize' : small_fontsize
}
# Dictionary of options for legend
opts_leg = {
    'size' : small_fontsize
}
# Dictionary of options for annotations
opts_text = {
    'fontsize' : small_fontsize
}

# Dictionary of options for markers
opts_mark = {
    'markersize' : ms,
    'markeredgewidth' : melw,
    'markeredgecolor' : melc
}
# Dictionary of options for lines
opts_line = {
    'linewidth' : lw
}
# Dictionary of alterantive options for lines
opts_line_a = opts_line.copy()
opts_line_a['linewidth'] = 2*lw

# Special options for markers
opts_mark_sp = {
    'markersize' : ms,
    'markeredgewidth' : 0,
    'markeredgecolor' : None
}

#%% Data import and assignment

# Import data
tab = pd.read_excel('pH_data.xlsx')
# Extract data
time = tab.loc[:, 'Time'].to_numpy()
Y_meas = tab.loc[:, 'Y'].to_numpy()
pH_meas = tab.loc[:, 'pH'].to_numpy()
u_meas = tab.loc[:, 'u'].to_numpy()
F_meas = tab.loc[:, 'F'].to_numpy()

# Assign data
X = Y_meas.copy().reshape(-1, 1)
y = pH_meas.copy().reshape(-1, 1)

# numbers of observations and variables
N, V = X.shape

# Collocation points for constraints
X_c = np.linspace(-0.1, 0.02, 2001, endpoint = True).reshape(-1, 1)

# Domain for plots
X_vals = np.linspace(-0.1, 0.06, 2001, endpoint = True).reshape(-1, 1)

# Plot data
ylabels = [
    '$\mathrm{Y\ (mol)}$',
    '$\mathrm{pH}$',
    '$\mathrm{u\ (L / min)}$',
    '$\mathrm{F\ (L / min)}$'
]
ylims = np.array([
    (-0.1, 0.025),
    (0, 15),
    (0, 0.15),
    (0, 1.5)
])
col_idx = [1, 1, 2, 4]

# Process variables
fig, ax = plt.subplots(nrows = 4, ncols = 1, sharex = True, figsize = large_fs)
for i in range(4):
    ax[i].plot(tab.iloc[:, 0], tab.iloc[:, i + 1], '.', markerfacecolor = colours[col_idx[i]], **opts_mark_sp)
    ax[i].set_ylabel(ylabels[i], **opts_labs)
    ax[i].tick_params(axis = 'both', **opts_ticks)
    ax[i].set_ylim(*ylims[i])
    ax[i].minorticks_on()
    ax[i].tick_params(axis = 'both', which = 'minor', **opts_ticks)
ax[i].set_xlim(0, 400)
ax[i].set_xlabel('Time (min)', **opts_labs)
plt.tight_layout()

# Plot titration curve
xlims = (-0.1, 0.06)
ylims = (0, 14)
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = small_fs)
ax.plot(X, y, 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_xlabel('$\mathrm{Y\ (mol)}$', **opts_labs)
ax.set_ylabel('$\mathrm{pH}$', **opts_labs)
ax.tick_params(axis = 'both', **opts_ticks)
ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 2))
ax.minorticks_on()
ax.tick_params(axis = 'both', which = 'minor', **opts_ticks)
leg = ax.legend(loc = 'lower left'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% Setting for ANN models

# Preprocessors for input and output data
preprocessor_X = StandardScaler(with_mean = True, with_std = True)
preprocessor_y = StandardScaler(with_mean = True, with_std = True)

# Number of inputs
N_i = np.shape(X)[1]
# Number of outputs
N_o = np.shape(y)[1]

# Number of neurons in the hidden layer
N_h = 9
# Activation function of the hidden layer
f_h = tanh
# Activation function of the output layer
f_o = identity
# Gather activation functions
f = [f_h, f_o]

# Number of computational layers
L = len(f)
# Size of the layers
Ns = [N_i, N_h, N_o]
# Total number of parameters
Np = np.dot((1 + np.array(Ns[0:-1])), np.array(Ns[1:]))

# Random numebr generator
rseed = np.random.RandomState(seed = 20240731)
rng = np.random.default_rng(20240731)
# Intialize parameters
p_0 = rng.standard_normal(Np)

# Optimization options
opt = {
        'disp' : False,
        'gtol' : 1e-3
}

# Prepare list of model performances
mod_perf = {
    'model' : [],
    'training' : []
}

#%% Data preparation

# Preprocess data
preprocessor_X = preprocessor_X.fit(X)
preprocessor_y = preprocessor_y.fit(y)
X_s = preprocessor_X.transform(X)
y_s = preprocessor_y.transform(y)

# Scale collocation points
X_cs = preprocessor_X.transform(X_c)

# Lower and upper bounds for the output
y_lb = preprocessor_y.transform(np.array([[2]]))
y_ub = preprocessor_y.transform(np.array([[13]]))

# Numerical derivative operator
diff_fun = fd.FinDiff(1, (X_cs[1] - X_cs[0])[0], 1, acc = 2)

# First prinicple model
def fpm_pH(Y, K_w = 1e-14):
    '''
    First principles model of the pH vs. Y relationship
    '''
    return -np.log10(0.5*(Y + np.sqrt(Y**2 + 4*K_w)))

# Compute reference pH values from FPM
y_ref = preprocessor_y.transform(fpm_pH(X_c))

#%% Simple ANN model

# Solve optimization problem with gradients
OPT = minimize(
    ANN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f),
    method = 'BFGS',
    jac = ANN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_ANN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)

# Performance metric
PM_ANN = MSE(y.flatten(), y_pred_ANN.flatten())

# Save model performance
mod_perf['model'].append('ANN')
mod_perf['training'].append(PM_ANN)

# Display results
print('%--------> ANN <--------%')
print('Performance metric:', PM_ANN, end = '\n\n')

# Apply the model for plot
y_pred_ANN_vals = preprocessor_y.inverse_transform(
    ANN(
        preprocessor_X.transform(X_vals).T, W, b, f
    ).T
)

# Plot model fit
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = small_fs)
ax.plot(X, y, 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
# ax.plot(X_c, np.full(X_c.shape, 0), '|', label = 'Collocation points', markeredgecolor = colours[-1])
ax.plot(X_vals, y_pred_ANN_vals, '-', label = 'ANN', color = colours[3], **opts_line_a)
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_xlabel('$\mathrm{Y\ (mol)}$', **opts_labs)
ax.set_ylabel('$\mathrm{pH}$', **opts_labs)
ax.tick_params(axis = 'both', **opts_ticks)
ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 2))
ax.minorticks_on()
ax.tick_params(axis = 'both', which = 'minor', **opts_ticks)
leg = ax.legend(loc = 'lower left'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN with upper bound constraints

# Weights of the mechanistic consitraint
alpha = np.array([
    0.0,   # Lower bound
    0.1,   # Upper bound
    0.0,   # Monotonicity
    0.0    # Mechanistic model equaility
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, None, y_ub, None, None),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p_PGNN = OPT_PGNN.x
# Reconstruct parameter matrices
W_PGNN, b_PGNN = ANN_params_to_matrices(p_PGNN, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W_PGNN, b_PGNN, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())

# Save model performance
mod_perf['model'].append('PGNN_a')
mod_perf['training'].append(PM_PGNN)

# Display results
print('%--------> PGNN_a (upper bound constraint) <--------%')
print('Performance metric:', PM_PGNN, end = '\n\n')

# Apply the model for plot
y_pred_PGNN_vals = preprocessor_y.inverse_transform(
    ANN(
        preprocessor_X.transform(X_vals).T, W_PGNN, b_PGNN, f
    ).T
)

# Plot model fit
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = small_fs)
ax.plot(X, y, 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
ax.plot(X_c, np.full(X_c.shape, 0), '|', label = 'Collocation points', markeredgecolor = colours[-1])
ax.plot(X_vals, y_pred_ANN_vals, '-', label = 'ANN', color = colours[3], **opts_line_a)
ax.plot(X_vals, y_pred_PGNN_vals, '-', label = 'PGNN_a', color = colours[4], **opts_line)
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_xlabel('$\mathrm{Y\ (mol)}$', **opts_labs)
ax.set_ylabel('$\mathrm{pH}$', **opts_labs)
ax.tick_params(axis = 'both', **opts_ticks)
ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 2))
ax.minorticks_on()
ax.tick_params(axis = 'both', which = 'minor', **opts_ticks)
leg = ax.legend(loc = 'lower left'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN with upper bound and monotonicity constraints

# Weights of the mechanistic consitraint
alpha = np.array([
    0.0,   # Lower bound
    0.1,   # Upper bound
    0.1,   # Monotonicity
    0.0    # Mechanistic model equaility
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, None, y_ub, diff_fun, None),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p_PGNN = OPT_PGNN.x
# Reconstruct parameter matrices
W_PGNN, b_PGNN = ANN_params_to_matrices(p_PGNN, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W_PGNN, b_PGNN, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())

# Save model performance
mod_perf['model'].append('PGNN_b')
mod_perf['training'].append(PM_PGNN)

# Display results
print('%--------> PGNN_b (monotonicity constraint) <--------%')
print('Performance metric:', PM_PGNN, end = '\n\n')

# Apply the model for plot
y_pred_PGNN_vals = preprocessor_y.inverse_transform(
    ANN(
        preprocessor_X.transform(X_vals).T, W_PGNN, b_PGNN, f
    ).T
)

# Plot model fit
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = small_fs)
ax.plot(X, y, 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
ax.plot(X_c, np.full(X_c.shape, 0), '|', label = 'Collocation points', markeredgecolor = colours[-1])
ax.plot(X_vals, y_pred_ANN_vals, '-', label = 'ANN', color = colours[3], **opts_line_a)
ax.plot(X_vals, y_pred_PGNN_vals, '-', label = 'PGNN_b', color = colours[4], **opts_line)
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_xlabel('$\mathrm{Y\ (mol)}$', **opts_labs)
ax.set_ylabel('$\mathrm{pH}$', **opts_labs)
ax.tick_params(axis = 'both', **opts_ticks)
ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 2))
ax.minorticks_on()
ax.tick_params(axis = 'both', which = 'minor', **opts_ticks)
leg = ax.legend(loc = 'lower left'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN with both bounds and monotonicity constraints

# Weights of the mechanistic consitraint
alpha = np.array([
    0.1,   # Lower bound
    0.1,   # Upper bound
    0.1,   # Monotonicity
    0.0    # Mechanistic model equaility
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, y_lb, y_ub, diff_fun, None),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p_PGNN = OPT_PGNN.x
# Reconstruct parameter matrices
W_PGNN, b_PGNN = ANN_params_to_matrices(p_PGNN, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W_PGNN, b_PGNN, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())

# Save model performance
mod_perf['model'].append('PGNN_c')
mod_perf['training'].append(PM_PGNN)

# Display results
print('%--------> PGNN_c (bounds and monotonicity constraints) <--------%')
print('Performance metric:', PM_PGNN, end = '\n\n')

# Apply the model for plot
y_pred_PGNN_vals = preprocessor_y.inverse_transform(
    ANN(
        preprocessor_X.transform(X_vals).T, W_PGNN, b_PGNN, f
    ).T
)

# Plot model fit
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = small_fs)
ax.plot(X, y, 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
ax.plot(X_c, np.full(X_c.shape, 0), '|', label = 'Collocation points', markeredgecolor = colours[-1])
ax.plot(X_vals, y_pred_ANN_vals, '-', label = 'ANN', color = colours[3], **opts_line_a)
ax.plot(X_vals, y_pred_PGNN_vals, '-', label = 'PGNN_c', color = colours[4], **opts_line)
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_xlabel('$\mathrm{Y\ (mol)}$', **opts_labs)
ax.set_ylabel('$\mathrm{pH}$', **opts_labs)
ax.tick_params(axis = 'both', **opts_ticks)
ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 2))
ax.minorticks_on()
ax.tick_params(axis = 'both', which = 'minor', **opts_ticks)
leg = ax.legend(loc = 'lower left'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN with reference model constraint

# Weights of the mechanistic consitraint
alpha = np.array([
    2.5,   # Lower bound
    0.1,   # Upper bound
    0.2,   # Monotonicity
    0.2    # Mechanistic model equaility
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, y_lb, y_ub, diff_fun, y_ref.T),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p_PGNN = OPT_PGNN.x
# Reconstruct parameter matrices
W_PGNN, b_PGNN = ANN_params_to_matrices(p_PGNN, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W_PGNN, b_PGNN, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())

# Save model performance
mod_perf['model'].append('PGNN_d')
mod_perf['training'].append(PM_PGNN)

# Display results
print('%--------> PGNN_d (reference model constraint) <--------%')
print('Performance metric:', PM_PGNN, end = '\n\n')

# Apply the model for plot
y_pred_PGNN_vals = preprocessor_y.inverse_transform(
    ANN(
        preprocessor_X.transform(X_vals).T, W_PGNN, b_PGNN, f
    ).T
)

# Plot model fit
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = small_fs)
ax.plot(X, y, 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
ax.plot(X_c, np.full(X_c.shape, 0), '|', label = 'Collocation points', markeredgecolor = colours[-1])
ax.plot(X_vals, y_pred_ANN_vals, '-', label = 'ANN', color = colours[3], **opts_line_a)
ax.plot(X_vals, y_pred_PGNN_vals, '-', label = 'PGNN_d', color = colours[4], **opts_line)
ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_xlabel('$\mathrm{Y\ (mol)}$', **opts_labs)
ax.set_ylabel('$\mathrm{pH}$', **opts_labs)
ax.tick_params(axis = 'both', **opts_ticks)
ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 2))
ax.minorticks_on()
ax.tick_params(axis = 'both', which = 'minor', **opts_ticks)
leg = ax.legend(loc = 'lower left'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% Show summary of model performance

# Make dataframe
mod_perf_tab = pd.DataFrame(mod_perf)
# Convert MSE intro RMSE
mod_perf_tab.loc[:, 'training'] = np.sqrt(mod_perf_tab.loc[:, 'training'])
# Display dataframe
print('%--------> Summary of performance (RMSE of models) <--------%')
print(mod_perf_tab)