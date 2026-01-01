"""
Tesla Stock Prediction Project - Light LSTM Version

Objective:
Demonstrate a full pipeline of data acquisition, exploratory data analysis (EDA), 
data preprocessing, and a minimal LSTM model to predict 5-day average percentage change.

Key Points:
- Focus is on data cleaning and preprocessing, not predictive accuracy.
- Model is small and lightweight for demonstration purposes.
- All outputs (plots, CSVs) are saved in "results/light-lstm/" folder.
"""

# -----------------------------
# Import Libraries
# -----------------------------
import os              # For folder/file management
import numpy as np     # Numerical operations
import pandas as pd    # Data manipulation
import matplotlib.pyplot as plt  # Plotting
import yfinance as yf  # Download stock data
from sklearn.preprocessing import MinMaxScaler  # Scaling for LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from tensorflow.random import set_seed  # For reproducibility

# -----------------------------
# Reproducibility
# -----------------------------
# Fix random seeds so results are repeatable
np.random.seed(42)
set_seed(42)

# -----------------------------
# Setup Results Folder
# -----------------------------
# All outputs will be saved here
results_folder = "results/light-lstm"
os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# 1. Download Tesla Stock Data
# -----------------------------
# Fetch daily stock data from Yahoo Finance
# Period: 2015-01-01 to 2024-01-01
df = yf.download("TSLA", start="2015-01-01", end="2024-01-01", progress=False)

# Flatten multi-index columns if returned
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Print basic info
print("Number of rows and columns:", df.shape)
print(df.head())
print("Checking for null values:\n", df.isna().sum())

# -----------------------------
# 2. Exploratory Data Analysis (EDA)
# -----------------------------
# Summary statistics
eda_summary = df.describe()
eda_summary.to_csv(os.path.join(results_folder, "eda_summary.csv"))

# Plot stock price history
plt.figure(figsize=(12,6), dpi=150)
plt.plot(df["Open"], label="Open")
plt.plot(df["High"], label="High")
plt.plot(df["Low"], label="Low")
plt.plot(df["Close"], label="Close")
plt.title('Tesla Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(results_folder, "tsla_price_history.png"))
plt.show()

# Plot stock volume history
plt.figure(figsize=(12,6), dpi=150)
plt.plot(df["Volume"], label="Volume", color='purple')
plt.title('Tesla Stock Volume History')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(results_folder, "tsla_volume_history.png"))
plt.show()

# -----------------------------
# 3. Data Cleaning & Preprocessing
# -----------------------------
# Compute daily % change of Close price
df['Pct_Change'] = df['Close'].pct_change() * 100

# Compute 5-day average % change (target for model)
df['Target_5d_Avg'] = df['Pct_Change'].rolling(5).mean().shift(-5)

# Drop rows with NaN values created by rolling and shift
df = df.dropna()

# Scale Close prices to range [0,1] for LSTM input
scaler = MinMaxScaler(feature_range=(0,1))
close_scaled = scaler.fit_transform(df[['Close']])

# Prepare sequences for LSTM
sequence_length = 60  # use last 60 days to predict next 5-day avg % change
X, y = [], []

for i in range(sequence_length, len(close_scaled)):
    X.append(close_scaled[i-sequence_length:i, 0])  # input sequence
    y.append(df['Target_5d_Avg'].values[i])         # target value

# Convert to numpy arrays
X, y = np.array(X), np.array(y)

# Reshape X for LSTM [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)  # e.g., (number_of_samples, 60, 1)
print("y shape:", y.shape)  # e.g., (number_of_samples, )

# Train/test split (75% train, 25% test)
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# 4. Minimal LSTM Model
# -----------------------------
# Small LSTM network for demonstration purposes
model = Sequential()
model.add(LSTM(30, input_shape=(X_train.shape[1],1)))  # 30 units, simple architecture
model.add(Dense(1))  # single output for regression

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# EarlyStopping callback to avoid overfitting and save training time
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,           # small number of epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6. Plot Training Loss
# -----------------------------
plt.figure(figsize=(10,6), dpi=150)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Demo LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(results_folder, "loss_plot.png"))
plt.show()

# -----------------------------
# 7. Make Predictions and Plot
# -----------------------------
predictions = model.predict(X_test)

plt.figure(figsize=(12,6), dpi=150)
plt.plot(y_test, color='blue', label='Actual 5-day Avg % Change')
plt.plot(predictions, color='red', label='Predicted 5-day Avg % Change')
plt.title('Tesla 5-Day Avg % Change Prediction (Demo Model)')
plt.xlabel('Time Step')
plt.ylabel('% Change')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(results_folder, "tsla_5d_avg_prediction.png"))
plt.show()

# -----------------------------
# 8. Save Predictions to CSV
# -----------------------------
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions.flatten()
})
results_df.to_csv(os.path.join(results_folder, "predictions.csv"), index=False)

print(f" FDS project : Demo modeling complete. All results saved in '{results_folder}/'")
