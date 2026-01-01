import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Setup results folder
# -----------------------------
os.makedirs("results/lstm-full", exist_ok=True)

# -----------------------------
# Fix np.object warning for older libraries
# -----------------------------
if not hasattr(np, "object"):
    np.object = object

# -----------------------------
# Download Tesla stock data
# -----------------------------
df = yf.download(
    "TSLA",
    start="2015-01-01",
    end="2024-01-01",  # updated to 2024
    progress=False
)

# Flatten multiindex columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

print('Number of rows and columns:', df.shape)
print(df.head())
print("Checking if any null values are present:\n", df.isna().sum())

# -----------------------------
# Plot stock prices
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df["Open"])
plt.plot(df["High"])
plt.plot(df["Low"])
plt.plot(df["Close"])
plt.title('Tesla Stock Price History')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.savefig("results/tsla_price_history.png")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df["Volume"])
plt.title('Tesla Stock Volume History')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.savefig("results/tsla_volume_history.png")
plt.show()

# -----------------------------
# Compute 5-day average % change (target)
# -----------------------------
df['Pct_Change'] = df['Close'].pct_change() * 100
df['Target_5d_Avg'] = df['Pct_Change'].rolling(5).mean().shift(-5)
df = df.dropna()

# -----------------------------
# Prepare LSTM training data
# -----------------------------
sequence_length = 180  # use last 180 days to predict next 5-day avg
close_scaled = MinMaxScaler().fit_transform(df[['Close']])
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

# -----------------------------
# Split train/test sets
# -----------------------------
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# Build LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))  # Regression output

model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------
# Train the model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,  # reduce from 100 for faster testing
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save training loss plot
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("results/loss_plot.png")
plt.show()

# -----------------------------
# Make predictions on test set
# -----------------------------
predictions = model.predict(X_test)
# Optional: inverse scaling is tricky because y is % change, so keep as is

plt.figure(figsize=(12,6))
plt.plot(y_test, color='blue', label='Actual 5-day Avg % Change')
plt.plot(predictions, color='red', label='Predicted 5-day Avg % Change')
plt.title('Tesla 5-Day Avg % Change Prediction')
plt.ylabel('% Change')
plt.xlabel('Time Step')
plt.legend()
plt.savefig("results/tsla_5d_avg_prediction.png")
plt.show()

