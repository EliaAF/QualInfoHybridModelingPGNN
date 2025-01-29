"""
CA_case_study.py
Version: 1.0.0
Date: 2024/12/19
Author: Elia Arnese-Feffin elia249@mit.edu

# GNU General Public License version 3 (GPL-3.0) ------------------------------

CA_case_study.py
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
from PGNN_prototype import relu, identity#, tanh_der, identity_der
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

# Number of rows and number of columns in plots
nc = 1
nr = 3
# Special smallfigure size
small_fs_sp = (small_fs[0], 3.6)

# Special options for markers
opts_mark_sp = {
    'markersize' : ms,
    'markeredgewidth' : 0,
    'markeredgecolor' : None
}

#%% Data import and assignment

# Import data
tab_all = pd.read_excel('deactivation_data.xlsx')

# Add a variable for total volume flow over each batch
tab_all['V_F'] = np.nan
# Get timestep for integral
Delta_t = tab_all['Time'][1]*24*3600
# Loop over batches
for i in range(7):
    # Find batch
    idx = tab_all['batch_number'] == i + 1
    # Compute total volume flown for each batch
    tab_all.loc[idx, 'V_F'] = np.cumsum(Delta_t*tab_all.loc[idx, 'F']).shift(periods = 1)

# Copy table
tab = tab_all.copy()

# Variables to shift (for steady state tabulation)
vts = ['c_A', 'c_B', 'c_C', 'c_D', 'T', 'T_j', 'a', 'X_A', 'X_B', 'Y_C_A', 'Y_D_A', 'S_C_D']
# Loop over batches
for i in range(7):
    # Find batch
    idx = tab['batch_number'] == i + 1
    # Shift states and outputs bacward by one sampling time
    tab.loc[idx, vts] = tab.loc[idx, vts].shift(periods = -1)

# Drop some batches
btd = [2, 3, 5, 6]
# And keep other batches
btk = [i + 1 for i in range(7) if i + 1 not in btd]
# Empty the batches to drop
for b in btd:
    idx = tab['batch_number'] == b
    tab.loc[idx, :] = np.nan
# and remove observations at time = 0
idx = tab['Time'] == 0
tab.loc[idx, :] = np.nan
# Drop batches
tab.dropna(inplace = True)
# Fix numbering of batches
for i, b in enumerate(btk):
    idx = tab['batch_number'] == b
    tab.loc[idx, 'batch_number'] = i + 1

# Variables to use as inputs
input_vars = [
    'F',
    'T_F',
    'T_j_F',
    'F_j',
    'c_A',
    'c_B',
    'c_C',
    'T',
    'T_j',
]
# Variable to use as output
output_vars = ['a']

# Extract data
idx = tab['training'] == 1
X = tab.loc[idx, input_vars].to_numpy()
y = tab.loc[idx, output_vars].to_numpy()
idx = tab['testing'] == 1
X_test = tab.loc[idx, input_vars].to_numpy()
y_test = tab.loc[idx, output_vars].to_numpy()

# Numbers of observations and variables
N, V = X.shape

# Data plots
vtp = ['c_A_F', 'c_C_F', 'T_F', 'F', 'a', 'Y_C_A', 'T', 'F_j']
var_labs = [
    '$\mathrm{c_A^{in}\ (kmol / m^{3})}$',
    '$\mathrm{c_R^{in}\ (kmol / m^{3})}$',
    '$\mathrm{T^{in}\ (K)}$',
    '$\mathrm{F\ (m^{3} / s)}$',
    '$\mathrm{a}$',
    '$\mathrm{Y_{R/A}}$',
    '$\mathrm{T\ (K)}$',
    '$\mathrm{F_j\ (m^{3} / s)}$'
]
ylims = [
    [5.35, 5.45],
    [0, 0.002],
    [290, 310],
    [0.03, 0.07],
    [0, 1],
    [0.75, 0.95],
    [320, 380],
    [0, 0.1]
]
cv = [4, 4, 4, 4, 3, 1, 3, 2]

# Plot training batch
btp = 1
xlims = (0, 420)
idx = (tab['batch_number'] == btp)
fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize = large_fs, sharex = True)
for i in range(8):
    j = np.unravel_index(i, (4, 2), order = 'F')
    ax[j].plot(tab.loc[idx, 'Time'], tab.loc[idx, vtp[i]], 'o', label = 'Data', markerfacecolor = colours[cv[i]], **opts_mark_sp)
    ax[j].set_ylim(*ylims[i])
    ax[j].set_ylabel(var_labs[i], **opts_labs)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 60))
ax[1, 1].set_yticks(np.arange(0.75, 0.95 + 0.01, 0.05))
ax[2, 1].set_yticks(np.arange(320, 380 + 1, 20))
ax[3, 0].set_yticks(np.arange(0.03, 0.07 + 0.001, 0.01))
for i in range(2):
    ax[3, i].set_xlabel('Time (days)', **opts_labs)
fig.suptitle('Training batch', **opts_tit)
plt.tight_layout()

# Plot collocation batch
btp = 2
xlims = (0, 720)
idx = (tab['batch_number'] == btp)
fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize = large_fs, sharex = True)
for i in range(8):
    j = np.unravel_index(i, (4, 2), order = 'F')
    ax[j].plot(tab.loc[idx, 'Time'], tab.loc[idx, vtp[i]], 'o', label = 'Data', markerfacecolor = colours[cv[i]], **opts_mark_sp)
    ax[j].set_ylim(*ylims[i])
    ax[j].set_ylabel(var_labs[i], **opts_labs)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 120))
ax[1, 1].set_yticks(np.arange(0.75, 0.95 + 0.01, 0.05))
ax[2, 1].set_yticks(np.arange(320, 380 + 1, 20))
ax[3, 0].set_yticks(np.arange(0.03, 0.07 + 0.001, 0.01))
for i in range(2):
    ax[3, i].set_xlabel('Time (days)', **opts_labs)
fig.suptitle('Collocation points batch', **opts_tit)
plt.tight_layout()

# Plot testing batch
btp = 3
xlims = (0, 750)
idx = (tab['batch_number'] == btp)
fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize = large_fs, sharex = True)
for i in range(8):
    j = np.unravel_index(i, (4, 2), order = 'F')
    ax[j].plot(tab.loc[idx, 'Time'], tab.loc[idx, vtp[i]], 'o', label = 'Data', markerfacecolor = colours[cv[i]], **opts_mark_sp)
    ax[j].set_ylim(*ylims[i])
    ax[j].set_ylabel(var_labs[i], **opts_labs)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 150))
ax[1, 1].set_yticks(np.arange(0.75, 0.95 + 0.01, 0.05))
ax[2, 1].set_yticks(np.arange(320, 380 + 1, 20))
ax[3, 0].set_yticks(np.arange(0.03, 0.07 + 0.001, 0.01))
for i in range(2):
    ax[3, i].set_xlabel('Time (days)', **opts_labs)
fig.suptitle('Testing batch', **opts_tit)
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
N_h = 12
# Activation function of the hidden layer
f_h = relu
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
rseed = np.random.RandomState(seed = 20241203)
rng = np.random.default_rng(20241203)
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
    'training' : [],
    'testing' : []
}

#%% Data preparation

# Preprocess data
preprocessor_X = preprocessor_X.fit(X)
preprocessor_y = preprocessor_y.fit(y)
X_s = preprocessor_X.transform(X)
y_s = preprocessor_y.transform(y)

# Isolate all input data
X_all = preprocessor_X.transform(tab.loc[:, input_vars].to_numpy())

# Make dataframe for results
pred = tab.loc[:, ['Time', 'batch_number', 'training', 'collocation', 'testing', 'a']]

#%% Collocation points and reference values

# Functions for simplified first principles model
def fpm_CA_simplified(
        c,
        T,
        T_j,
        F,
        c_F,
        T_F,
        T_j_F,
        F_j,
        V = 1.28207,
        k_0 = 3.5346e9,
        Ea = 55e3,
        R_gas = 8.31446261815324,
        alpha_A = 1.3,
        alpha_B = 0.7,
        DeltaH = -23e3,
        rho_m = 900,
        Cp_m = 1.8,
        rho_w = 1000,
        Cp_w = 4.181
    ):
    '''
    First principles model of the catalyst deactivation case study: steady state
    material balance, explicit activity computation, simplified form
    '''
    R = k_0*np.exp(-Ea/(R_gas*T))*(c[0]**alpha_A)*(c[1]**alpha_B)
    div_fac = 1/(V*R)
    a_cA = F*(c_F[0] - c[0])*div_fac
    a_cB = F*(c_F[1] - c[1])*div_fac
    a_cC = -F*(c_F[2] - c[2])*div_fac
    a_T = (rho_m*Cp_m*F*(T_F - T) + rho_w*Cp_w*F_j*(T_j_F - T_j))*div_fac/DeltaH
    return np.stack((a_cA, a_cB, a_cC, a_T), axis = 1)
def fpm_CA_simplified_objective(
        x,
        a,
        c,
        T,
        T_j,
        F,
        c_F,
        T_F,
        T_j_F,
        F_j,
        scales,
        V = 1.28207,
        R_gas = 8.31446261815324,
        rho_m = 900,
        Cp_m = 1.8,
        rho_w = 1000,
        Cp_w = 4.181,
    ):
    '''
    Objective function to fit the simplified first principles model of the
    catalyst deactivation case study
    '''
    a_all = fpm_CA_simplified(c, T, T_j, F, c_F, T_F, T_j_F, F_j,
            k_0 = x[0]*scales[0],
            Ea = x[1]*scales[1],
            alpha_A = x[2]*scales[2],
            alpha_B = x[3]*scales[3],
            DeltaH = x[4]*scales[4]
    )
    return np.mean((a_all - a)**2)

# Select calibration data for mechanistic model fitting
idx = (tab['collocation'] == False) & (tab['testing'] == False)
tab_fpm = tab.loc[idx, :].copy()

# Get variables from table
a = tab_fpm['a'].to_numpy().reshape(-1, 1)
c = tab_fpm[['c_A', 'c_B', 'c_C', 'c_D']].to_numpy().T
T = tab_fpm['T'].to_numpy()
T_j = tab_fpm['T_j'].to_numpy()
F = tab_fpm['F'].to_numpy()
c_F = tab_fpm[['c_A_F', 'c_B_F', 'c_C_F', 'c_D_F']].to_numpy().T
T_F = tab_fpm['T_F'].to_numpy()
T_j_F = tab_fpm['T_j_F'].to_numpy()
F_j = tab_fpm['F_j'].to_numpy()

# Scale factors for dimensionless variables
scales = np.array([
    1e9,    # k_0
    1e3,    # Ea
    1,      # alpha_A
    1,      # alpha_B
    -1e3    # DeltaH
])
# Initialize parameters to be identified
x_0 = np.array([
    10,#3e9,   # k_0
    10,#40e3,       # Ea
    1,        # alpha_A
    1,        # alpha_B
    10#-30e3       # DeltaH
])

# Callback function
Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1:2.6f}   {2:2.6f}   {3:2.6f}   {4:2.6f}   {5:2.6f}   {6:2.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], fpm_CA_simplified_objective(Xi, a, c, T, T_j, F, c_F, T_F, T_j_F, F_j, scales)))
    Nfeval += 1
strtp = 'N      O(N)       x_1         x_2        x_3        x_4        x_5'
# print(strtp)

# Solve optimization problem with gradients
OPT = minimize(
    fpm_CA_simplified_objective,
    x_0,
    args = (a, c, T, T_j, F, c_F, T_F, T_j_F, F_j, scales),
    method = 'BFGS',
    jac = 'cs',
    callback = None,#callbackF,
    options = {
            'disp' : False,
            'gtol' : 1e-8
    }
)
# Get results and print optimization output
# print(OPT)
x = OPT.x*scales

# Find batches
NB = len(np.unique(tab.batch_number))

# Select data fro collocation points
idx = tab['collocation'] == True
tab_c = tab.loc[idx, :].copy()

# Get variables from collocation points table
c = tab_c[['c_A', 'c_B', 'c_C', 'c_D']].to_numpy().T
T = tab_c['T'].to_numpy()
T_j = tab_c['T_j'].to_numpy()
F = tab_c['F'].to_numpy()
c_F = tab_c[['c_A_F', 'c_B_F', 'c_C_F', 'c_D_F']].to_numpy().T
T_F = tab_c['T_F'].to_numpy()
T_j_F = tab_c['T_j_F'].to_numpy()
F_j = tab_c['F_j'].to_numpy()

# Upper and lower bounds of activity
a_lb = preprocessor_y.transform(np.array([[0.15]]))
a_ub = preprocessor_y.transform(np.array([[1.0]]))

# Numerical derivative operator
diff_fun = fd.FinDiff(1, tab_c['V_F'].to_numpy()/np.diff(tab_c['V_F']).mean(), acc = 4)

# Run the reference model
a_ref = fpm_CA_simplified(c, T, T_j, F, c_F, T_F, T_j_F, F_j,
        k_0 = x[0],
        Ea = x[1],
        alpha_A = x[2],
        alpha_B = x[3],
        DeltaH = x[4]
)

# Also run the incorrect reference model
a_ref_inc = fpm_CA_simplified(c, T, T_j, F, c_F, T_F, T_j_F, F_j,
        k_0 = x[0],
        Ea = x[1],
        alpha_A = 1,
        alpha_B = 1,
        DeltaH = x[4]
)

# Scale reference activity
for i in range(a_ref.shape[1]):
    a_ref[:, i] = preprocessor_y.transform(a_ref[:, i].reshape(-1, 1)).flatten()
    a_ref_inc[:, i] = preprocessor_y.transform(a_ref_inc[:, i].reshape(-1, 1)).flatten()

# Get collocation point matrix
idx = tab['collocation'] == 1
X_c = tab.loc[idx, input_vars].to_numpy()
# Scale collocation points
X_cs = preprocessor_X.transform(X_c)

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
# Apply the ANN to testing data
y_pred_ANN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_ANN = MSE(y.flatten(), y_pred_ANN.flatten())
PM_ANN_test = MSE(y_test.flatten(), y_pred_ANN_test.flatten())

# Save model performance
mod_perf['model'].append('ANN')
mod_perf['training'].append(PM_ANN)
mod_perf['testing'].append(PM_ANN_test)

# Display results
print('%--------> ANN <--------%')
print('Performance metric (train):', PM_ANN, end = '\n')
print('Performance metric  (test):', PM_ANN_test, end = '\n\n')

# Apply ANN to all data
pred['a_ANN'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
xlims = (0, 750)
ylims = (0, 1)
titles = ['Training', 'Collocation points', 'Testing']
fig, ax = plt.subplots(nrows = nr - 1, ncols = nc, sharex = True, sharey = True, figsize = (small_fs_sp[0], small_fs_sp[1]*2.1/3))
for k, i in enumerate([0, 2]):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(k, (nr, nc))
    else:
        j = k
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 150))
if nc > 1:
    for c in range(nc):
        ax[nr - 1 - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr - 1):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1 - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr - 1):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()
    
#%% PGNN model (bound constraints)

# Weights of the mechanistic constraints
alpha = np.array([
    0.1,    # Lower bound
    0.1,    # Upper bound
    0.0,    # Monotonicity constraint
    0.00,   # Reference activity from material balance of A
    0.00,   # Reference activity from material balance of B
    0.00,   # Reference activity from material balance of C
    0.00    # Reference activity from energy balance of reactor
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, a_lb, a_ub, None, None),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT_PGNN.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)
# Apply the ANN to testing data
y_pred_PGNN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())
PM_PGNN_test = MSE(y_test.flatten(), y_pred_PGNN_test.flatten())

# Save model performance
mod_perf['model'].append('PGNN_a')
mod_perf['training'].append(PM_PGNN)
mod_perf['testing'].append(PM_PGNN_test)

# Display results
print('%--------> PGNN_a (bound contraints) <--------%')
print('Performance metric (train):', PM_PGNN, end = '\n')
print('Performance metric  (test):', PM_PGNN_test, end = '\n\n')

# Apply ANN to all data
pred['PGNN_a'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
fig, ax = plt.subplots(nrows = nr, ncols = nc, sharex = True, sharey = True, figsize = small_fs_sp)
for i in range(NB):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(i, (nr, nc))
    else:
        j = i
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'PGNN_a'], '-', label = 'PGNN_a', color = colours[4], **opts_line)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 75))
if nc > 1:
    for c in range(nc):
        ax[nr - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN model (bound and slope constraints)

# Weights of the mechanistic constraints
alpha = np.array([
    0.1,    # Lower bound
    0.1,    # Upper bound
    0.0175, # Monotonicity constraint
    0.00,   # Reference activity from material balance of A
    0.00,   # Reference activity from material balance of B
    0.00,   # Reference activity from material balance of C
    0.00    # Reference activity from energy balance of reactor
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, a_lb, a_ub, diff_fun, None),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT_PGNN.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)
# Apply the ANN to testing data
y_pred_PGNN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())
PM_PGNN_test = MSE(y_test.flatten(), y_pred_PGNN_test.flatten())

# Save model performance
mod_perf['model'].append('PGNN_b')
mod_perf['training'].append(PM_PGNN)
mod_perf['testing'].append(PM_PGNN_test)

# Display results
print('%--------> PGNN_b (bound and slope contraints) <--------%')
print('Performance metric (train):', PM_PGNN, end = '\n')
print('Performance metric  (test):', PM_PGNN_test, end = '\n\n')

# Apply ANN to all data
pred['PGNN_b'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
fig, ax = plt.subplots(nrows = nr, ncols = nc, sharex = True, sharey = True, figsize = small_fs_sp)
for i in range(NB):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(i, (nr, nc))
    else:
        j = i
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'PGNN_b'], '-', label = 'PGNN_b', color = colours[4], **opts_line)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 75))
if nc > 1:
    for c in range(nc):
        ax[nr - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()


#%% PGNN model (reference model constraints)

# Weights of the mechanistic constraints
alpha = np.array([
    0.1,    # Lower bound
    0.1,    # Upper bound
    0.0,    # Monotonicity constraint
    0.033,   # Reference activity from material balance of A
    0.033,   # Reference activity from material balance of B
    0.033,   # Reference activity from material balance of C
    0.00    # Reference activity from energy balance of reactor
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, a_lb, a_ub, None, a_ref.T),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT_PGNN.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)
# Apply the ANN to testing data
y_pred_PGNN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())
PM_PGNN_test = MSE(y_test.flatten(), y_pred_PGNN_test.flatten())

# Save model performance
mod_perf['model'].append('PGNN_c')
mod_perf['training'].append(PM_PGNN)
mod_perf['testing'].append(PM_PGNN_test)

# Display results
print('%--------> PGNN_c (reference model contraints) <--------%')
print('Performance metric (train):', PM_PGNN, end = '\n')
print('Performance metric  (test):', PM_PGNN_test, end = '\n\n')

# Apply ANN to all data
pred['PGNN_c'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
fig, ax = plt.subplots(nrows = nr, ncols = nc, sharex = True, sharey = True, figsize = small_fs_sp)
for i in range(NB):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(i, (nr, nc))
    else:
        j = i
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'PGNN_c'], '-', label = 'PGNN_c', color = colours[4], **opts_line)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 75))
if nc > 1:
    for c in range(nc):
        ax[nr - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN model (alternative reference model constraints)

alpha = np.array([
    0.1,    # Lower bound
    0.1,    # Upper bound
    0.0,    # Monotonicity constraint
    0.00,   # Reference activity from material balance of A
    0.00,   # Reference activity from material balance of B
    0.00,   # Reference activity from material balance of C
    0.1     # Reference activity from energy balance of reactor
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, a_lb, a_ub, None, a_ref.T),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT_PGNN.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)
# Apply the ANN to testing data
y_pred_PGNN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())
PM_PGNN_test = MSE(y_test.flatten(), y_pred_PGNN_test.flatten())

# Save model performance
mod_perf['model'].append('PGNN_d')
mod_perf['training'].append(PM_PGNN)
mod_perf['testing'].append(PM_PGNN_test)

# Display results
print('%--------> PGNN_d (alternative reference model contraints) <--------%')
print('Performance metric (train):', PM_PGNN, end = '\n')
print('Performance metric  (test):', PM_PGNN_test, end = '\n\n')

# Apply ANN to all data
pred['PGNN_d'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
fig, ax = plt.subplots(nrows = nr, ncols = nc, sharex = True, sharey = True, figsize = small_fs_sp)
for i in range(NB):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(i, (nr, nc))
    else:
        j = i
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'PGNN_d'], '-', label = 'PGNN_d', color = colours[4], **opts_line)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 75))
if nc > 1:
    for c in range(nc):
        ax[nr - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN model (all reference model constraints)

alpha = np.array([
    0.1,    # Lower bound
    0.1,    # Upper bound
    0.0,    # Monotonicity constraint
    0.025,  # Reference activity from material balance of A
    0.025,  # Reference activity from material balance of B
    0.025,  # Reference activity from material balance of C
    0.025   # Reference activity from energy balance of reactor
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, a_lb, a_ub, None, a_ref.T),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT_PGNN.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)
# Apply the ANN to testing data
y_pred_PGNN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())
PM_PGNN_test = MSE(y_test.flatten(), y_pred_PGNN_test.flatten())

# Save model performance
mod_perf['model'].append('PGNN_e')
mod_perf['training'].append(PM_PGNN)
mod_perf['testing'].append(PM_PGNN_test)

# Display results
print('%--------> PGNN_e (all reference model contraints) <--------%')
print('Performance metric (train):', PM_PGNN, end = '\n')
print('Performance metric  (test):', PM_PGNN_test, end = '\n\n')

# Apply ANN to all data
pred['PGNN_e'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
fig, ax = plt.subplots(nrows = nr, ncols = nc, sharex = True, sharey = True, figsize = small_fs_sp)
for i in range(NB):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(i, (nr, nc))
    else:
        j = i
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'PGNN_e'], '-', label = 'PGNN_e', color = colours[4], **opts_line)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 75))
if nc > 1:
    for c in range(nc):
        ax[nr - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% PGNN model (incorrect reference model constraints)

# Weights of the mechanistic constraints
alpha = np.array([
    0.1,    # Lower bound
    0.1,    # Upper bound
    0.0,    # Monotonicity constraint
    0.025,   # Reference activity from material balance of A
    0.025,   # Reference activity from material balance of B
    0.025,   # Reference activity from material balance of C
    0.025    # Reference activity from energy balance of reactor
])

# Solve optimization problem with gradients
OPT_PGNN = minimize(
    PGNN_objective,
    p_0,
    args = (X_s.T, y_s.T, L, Ns, Np, f, alpha, X_cs.T, a_lb, a_ub, None, a_ref_inc.T),
    method = 'BFGS',
    jac = PGNN_objective_gradient,
    options = opt
)

# Recover the parameteters
p = OPT_PGNN.x
# Reconstruct parameter matrices
W, b = ANN_params_to_matrices(p, L, Ns)
# Apply the ANN
y_pred_PGNN = preprocessor_y.inverse_transform(ANN(X_s.T, W, b, f).T)
# Apply the ANN to testing data
y_pred_PGNN_test = preprocessor_y.inverse_transform(ANN(preprocessor_X.transform(X_test).T, W, b, f).T)

# Performance metric
PM_PGNN = MSE(y.flatten(), y_pred_PGNN.flatten())
PM_PGNN_test = MSE(y_test.flatten(), y_pred_PGNN_test.flatten())

# Save model performance
mod_perf['model'].append('PGNN_f')
mod_perf['training'].append(PM_PGNN)
mod_perf['testing'].append(PM_PGNN_test)

# Display results
print('%--------> PGNN_f (incorrect reference model contraints) <--------%')
print('Performance metric (train):', PM_PGNN, end = '\n')
print('Performance metric  (test):', PM_PGNN_test, end = '\n\n')

# Apply ANN to all data
pred['PGNN_f'] = preprocessor_y.inverse_transform(ANN(X_all.T, W, b, f).T)

#% Plot

# Plot model fit
fig, ax = plt.subplots(nrows = nr, ncols = nc, sharex = True, sharey = True, figsize = small_fs_sp)
for i in range(NB):
    idx = pred['batch_number'] == i + 1
    if nc > 1:
        j = np.unravel_index(i, (nr, nc))
    else:
        j = i
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a'], 'o', label = 'Data', markerfacecolor = colours[0], **opts_mark)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'a_ANN'], '-', label = 'ANN', color = colours[3], **opts_line_a)
    ax[j].plot(pred.loc[idx, 'Time'], pred.loc[idx, 'PGNN_f'], '-', label = 'PGNN_f', color = colours[4], **opts_line)
    ax[j].text(0.03, 0.06, titles[i], transform = ax[j].transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', **opts_text)
    ax[j].minorticks_on()
    ax[j].tick_params(axis = 'both', which = 'both', **opts_ticks)
ax[j].set_xlim(*xlims)
ax[j].set_ylim(*ylims)
ax[j].set_xticks(np.arange(0, xlims[1] + 1, 75))
if nc > 1:
    for c in range(nc):
        ax[nr - 1, c].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r, 0].set_ylabel('a', **opts_labs)
    leg = ax[0, nc - 1].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
else:
    ax[nr - 1].set_xlabel('Time (days)', **opts_labs)
    for r in range(nr):
        ax[r].set_ylabel('a', **opts_labs)
    leg = ax[0].legend(loc = 'upper right'); plt.setp(leg.texts, **opts_leg)
plt.tight_layout()

#%% Save summary of model performance

# Make dataframe
mod_perf_tab = pd.DataFrame(mod_perf)
# Convert MSE intro RMSE
mod_perf_tab.loc[:, ['training', 'testing']] = np.sqrt(mod_perf_tab.loc[:, ['training', 'testing']])
# Display dataframe
print('%--------> Summary of performance (RMSE of models) <--------%')
print(mod_perf_tab)