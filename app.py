import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta


# ============================== #
# 🔹 Streamlit App Header        #
# ============================== #
st.title("📈 Stock Price Prediction App")
st.sidebar.header("Select Stock & Parameters")

# ============================== #
# 🔹 User Input: Select Stock    #
# ============================== #
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-02-25"))

# New parameter for future prediction days
future_days = st.sidebar.slider("Number of Days to Predict into Future:", 1, 30, 7)

# ============================== #
# 🔹 Fetch Stock Data            #
# ============================== #
st.write(f"Fetching data for **{ticker}** from {start_date} to {end_date}...")
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("Error fetching data. Please check the stock ticker.")
    st.stop()

data["Date"] = data.index
st.write(data.tail())

# ============================== #
# 🔹 Data Preprocessing          #
# ============================== #
prices = data["Close"].values
scaler = StandardScaler()
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Function to create lagged features
def create_lagged_data(prices, lag=5):
    X, y = [], []
    for i in range(len(prices) - lag):
        X.append(prices[i : i + lag])
        y.append(prices[i + lag])
    return np.array(X), np.array(y)

lag = 5
X, y = create_lagged_data(prices_scaled.flatten(), lag)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ============================== #
# 🔹 Define PyTorch Model        #
# ============================== #
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = StockPredictor(lag)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ============================== #
# 🔹 Train Model                 #
# ============================== #
epochs = 100
progress_bar = st.sidebar.progress(0)
loss_placeholder = st.sidebar.empty()

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        loss_placeholder.text(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        progress_bar.progress((epoch + 1) / epochs)

# ============================== #
# 🔹 Predict on Test Set         #
# ============================== #
y_pred_train = model(X_train_tensor).detach().numpy()
y_pred_test = model(X_test_tensor).detach().numpy()

# Convert back to actual prices
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_pred_train_actual = scaler.inverse_transform(y_pred_train).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_test_actual = scaler.inverse_transform(y_pred_test).flatten()

# ============================== #
# 🔹 Future Predictions          #
# ============================== #
st.subheader("🔮 Future Price Prediction")
# Get the most recent data points for prediction
last_sequence = prices_scaled[-lag:].flatten()

# Generate predictions for future days
future_predictions_scaled = []
current_sequence = last_sequence.copy()

for _ in range(future_days):
    # Convert to tensor for prediction
    current_tensor = torch.tensor(current_sequence.reshape(1, -1), dtype=torch.float32)
    # Make prediction
    next_pred = model(current_tensor).item()
    # Add to predictions
    future_predictions_scaled.append(next_pred)
    # Update sequence (remove oldest, add newest)
    current_sequence = np.append(current_sequence[1:], next_pred)

# Convert scaled predictions back to actual prices
future_predictions = scaler.inverse_transform(
    np.array(future_predictions_scaled).reshape(-1, 1)
).flatten()

# Create future dates for plotting
last_date = data.index[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]

# Display future predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_predictions
})
st.write("Future Price Predictions:")
st.write(future_df)

# ============================== #
# 🔹 Model Evaluation            #
# ============================== #
mse_test = mean_squared_error(y_test_actual, y_pred_test_actual)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_actual, y_pred_test_actual)
r2_test = r2_score(y_test_actual, y_pred_test_actual)

st.subheader("🔹 Model Performance Metrics")
st.write(f"📉 **MSE**: {mse_test:.4f}")
st.write(f"📉 **RMSE**: {rmse_test:.4f}")
st.write(f"📉 **MAE**: {mae_test:.4f}")
st.write(f"📈 **R² Score**: {r2_test:.4f}")

# ============================== #
# 🔹 Plot Predictions            #
# ============================== #
st.subheader("📊 Actual vs. Predicted Prices")
fig, ax = plt.subplots(figsize=(12, 6))

# Create proper date indices for plotting
# Account for the lag in training data
train_dates = data.index[lag:train_size+lag]
test_dates = data.index[train_size+lag:train_size+lag+len(y_test)]

# Plot with aligned indices
ax.plot(train_dates, y_train_actual, label="Actual (Train)", color="blue", alpha=0.7)
ax.plot(train_dates, y_pred_train_actual, label="Predicted (Train)", color="red", alpha=0.7)
ax.plot(test_dates, y_test_actual, label="Actual (Test)", color="green")
ax.plot(test_dates, y_pred_test_actual, label="Predicted (Test)", color="orange")

# Add future predictions with different style
ax.plot(future_dates, future_predictions, 'o-', label="Future Predictions", color="purple", linewidth=2)

# Add vertical line to separate historical data from future predictions
ax.axvline(x=data.index[-1], color='gray', linestyle='--', alpha=0.7)
ax.text(data.index[-1], ax.get_ylim()[0], 'Today', ha='right', va='bottom', alpha=0.7)

ax.legend()
ax.set_title(f"{ticker} Stock Prediction with {future_days} Day Future Forecast")
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()  # Rotate date labels for better readability
st.pyplot(fig)

# ============================== #
# 🔹 Confidence Disclaimer       #
# ============================== #
st.warning("""
**Disclaimer**: Future price predictions are based on historical patterns and may not account for unpredictable market events, 
news announcements, or economic changes. These predictions should be used for educational purposes only.
""")