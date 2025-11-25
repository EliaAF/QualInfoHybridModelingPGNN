"""
tensorflow_PGNN.py
Version: 0.8.0
Date: 2025/01/28
Author: Nidhish Sagar nidhishs@mit.edu

# GNU General Public License version 3 (GPL-3.0) ------------------------------

tensorflow_PGNN.py
Copyright (C) 2024-2025 Nidhish Sagar

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
    E. Arnese-Feffin, N. Sagar, L.A. Briceno-Mena, B. Braun, I. Castillo, C. Rizzo, L. Bui, J. Xu, L.H. Chiang, and R.D. Braatz (2026):
        The Incorporation of Qualitative Knowledge in Hybrid Modeling.
        Computers and Chemical Engineering, 205, 109484.
        DOI: https://doi.org/10.1016/j.compchemeng.2025.109484.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
file_path = 'pH_data.xlsx' # Change this path to your local directory
sheet_name = 'Sheet1' # Change this to the sheet name in your Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

# Extract time, x, and pH data
time_data = df.iloc[1:, 0]
t = time_data.to_numpy()

x_data = df.iloc[1:, 1]
x = x_data.to_numpy().reshape(-1, 1)

pH_meas_data = df.iloc[1:, 2]
pH_meas = pH_meas_data.to_numpy().reshape(-1, 1)

# Scale data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(x)
scaler_y = StandardScaler()
pH_meas_scaled = scaler_y.fit_transform(pH_meas)

# Convert X_scaled to TensorFlow tensor
X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)

# Define custom loss function with equality and inequality constraints
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compute gradient d(pH)/dx
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        y_pred_tape = model(X_tensor)
    dy_dx = tape.gradient(y_pred_tape, X_tensor)  # This is using in-built tensor gradient operator on the model, does NOT do finite differences
    
    # Penalty for violating d(pH)/dx < 0
    inequality_penalty = tf.reduce_mean(tf.maximum(dy_dx, 0.0))
    
    # Compute the equality constraint expression only for the current batch
    X_batch = tape.watched_variables()[0]
    
    # Safeguard for equality expression (prevent log of non-positive numbers)
    sqrt_expr = tf.math.sqrt(tf.square(X_batch) + 4e-14)  
    #tf.print("sqrt_expr:", sqrt_expr)
    equality_expr = -tf.math.log(tf.maximum((X_batch + sqrt_expr) / 2, 1e-6))  # add small value to prevent log(0)
    #tf.print("equality_expr:", equality_expr)
    #equality_penalty = tf.reduce_mean(tf.abs(y_pred - equality_expr)) 
    equality_penalty = tf.reduce_mean(y_pred - equality_expr)
    eq_weight = 1  # tunable based on requirement

    # Penalty for pH values outside the range [2, 13]
    lower_bound_penalty = tf.reduce_mean(tf.maximum(2.0 - y_pred, 0.0))
    upper_bound_penalty = tf.reduce_mean(tf.maximum(y_pred - 13.0, 0.0))
    bound_penalty = lower_bound_penalty + upper_bound_penalty
    
    # Total loss
    total_loss = mse_loss + inequality_penalty + (eq_weight * equality_penalty) + bound_penalty  
    
    # Print loss components for debugging
    #print("sqrt_expr:", sqrt_expr)
    #tf.print("MSE Loss:", mse_loss)
    #tf.print("Inequality Penalty:", inequality_penalty)
    #tf.print("Equality Penalty:", equality_penalty)
    #tf.print("Bound Penalty:", bound_penalty)
    
    return total_loss

# Build model
model = Sequential([
    Input(shape=(X_scaled.shape[1],)),  # Input layer
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for pH measurement
])

# Compile model with custom loss
model.compile(optimizer='adam', loss=custom_loss)

# Train model
model.fit(X_tensor, pH_meas_scaled, epochs=100, batch_size=16, validation_data=(X_tensor, pH_meas_scaled))

# Evaluate model
loss = model.evaluate(X_tensor, pH_meas_scaled)
print(f'Test loss: {loss}')

# Predict and inverse transform predictions
pH_pred_scaled = model.predict(X_tensor)
pH_pred = scaler_y.inverse_transform(pH_pred_scaled)

# Print predicted values for debugging
print("Predicted pH values:", pH_pred.flatten())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, pH_meas, 'b', label='Original pH Data')
plt.plot(x, pH_pred, 'r--', label='Neural Network Fit')
plt.xlabel('Time')
plt.ylabel('pH')
plt.title('pH Data and Neural Network Fit with Constraints')
plt.legend()
plt.grid(True)
plt.show()
