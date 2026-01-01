# Tesla Stock Prediction Project
# Predict 5-day average % change using LSTM
# Teacher grading points: Problem Definition, EDA, Preprocessing, Modeling, Results

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.random import set_seed

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(42)
set_seed(42)

# -----------------------------
# Setup results folder
# -----------------------------
os.makedirs("results/main", exist_ok=True)

# -----------------------------
# Fix np.object warning for older libraries
# -----------------------------
if not hasattr(np, "object"):
    np.object = object

# -----------------------------
# 1. Problem Definition & Objectives
# -----------------------------
print("""
Problem Definition:
Predict Tesla's short-term stock trend by forecasting the 5-day average percentage change.
Objective:
Use historical Tesla stock data (2015-2024) to train an LSTM deep learning model for regression.
""")

# -----------------------------
# 2. Download Data
# -----------------------------
df = yf.download(
    "TSLA",
    start="2022-01-01",
    end="2025-01-01",
    progress=False
)

# Flatten multiindex columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Quick sanity checks
print("Number of rows and columns:", df.shape)
print(df.head())
print("Checking for null values:\n", df.isna().sum())

# -----------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------

# Summary statistics
eda_summary = df.describe()
eda_summary.to_csv("results/eda_summary.csv")
print(eda_summary)

# Plot price history
plt.figure(figsize=(12,6))
plt.plot(df["Open"], label="Open")
plt.plot(df["High"], label="High")
plt.plot(df["Low"], label="Low")
plt.plot(df["Close"], label="Close")
plt.title('Tesla Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig("results/tsla_price_history.png")
plt.show()

# Plot volume history
plt.figure(figsize=(12,6))
plt.plot(df["Volume"], label="Volume", color='purple')
plt.title('Tesla Stock Volume History')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.savefig("results/tsla_volume_history.png")
plt.show()

# -----------------------------
# 4. Data Cleaning & Preprocessing
# -----------------------------
# Compute 5-day average % change as target
df['Pct_Change'] = df['Close'].pct_change() * 100
df['Target_5d_Avg'] = df['Pct_Change'].rolling(5).mean().shift(-5)
df = df.dropna()

# Prepare sequences for LSTM
sequence_length = 180  # last 180 days as features
scaler = MinMaxScaler(feature_range=(0,1))
close_scaled = scaler.fit_transform(df[['Close']])

X = []
y = []

for i in range(sequence_length, len(close_scaled)):
    X.append(close_scaled[i-sequence_length:i, 0])
    y.append(df['Target_5d_Avg'].values[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Train/test split
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# 5. Build LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))  # regression output

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# -----------------------------
# 6. Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save training loss plot
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("results/loss_plot.png")
plt.show()

# -----------------------------
# 7. Make Predictions
# -----------------------------
predictions = model.predict(X_test)

plt.figure(figsize=(12,6))
plt.plot(y_test, color='blue', label='Actual 5-day Avg % Change')
plt.plot(predictions, color='red', label='Predicted 5-day Avg % Change')
plt.title('Tesla 5-Day Avg % Change Prediction')
plt.xlabel('Time Step')
plt.ylabel('% Change')
plt.legend()
plt.savefig("results/tsla_5d_avg_prediction.png")
plt.show()

# -----------------------------
# 8. Save Results
# -----------------------------
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions.flatten()
})
results_df.to_csv("results/predictions.csv", index=False)

print("All results and plots saved in the 'results/' folder.")
