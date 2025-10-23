#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Data_Prophets_BiLstm_and_DeepAR.ipynb
Conversion Date: 2025-10-23T03:49:42.252Z
"""

import pandas as pd
df = pd.read_csv('cluster_0.csv')
df

pip install git+https://github.com/amazon-science/chronos-forecasting.git

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error

# Load and prepare data
df = pd.read_csv("cluster_0.csv")
df = df.drop(columns=["Unnamed: 0", "cluster"])
df = df.set_index("account").T
df.index = pd.to_datetime(df.index)
avg_usage = df.mean(axis=1)

# Time splits
train = avg_usage[:'2014-06-30']
val = avg_usage['2014-07-01':'2014-07-31']
test = avg_usage['2014-08-01':'2014-08-30']  # Target forecast horizon = 30 days

# Prepare context: last 60 days from train+val
context = pd.concat([train, val])[-60:].values
context_tensor = torch.tensor(context, dtype=torch.float).unsqueeze(0)  # shape: [1, 60]

# Load Chronos pipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="auto",
    torch_dtype=torch.float32  # Use bfloat16 if supported
)

# Predict next 30 time steps
prediction_length = 30
forecast_tensor = pipeline.predict(context_tensor, prediction_length=prediction_length)
forecast_values = forecast_tensor[0].tolist()

# Evaluate (fixing shape mismatch)
true_values = test.values
forecast_values = forecast_values[:len(true_values)]  # Truncate if forecast is longer
forecast_values
# mape = mean_absolute_percentage_error(true_values, forecast_values)
# print("MAPE:", mape)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(avg_usage.index, avg_usage, label="Actual")
plt.plot(test.index[:len(forecast_values)], forecast_values, label="Forecast", linestyle="--")
plt.axvspan(train.index[-1], test.index[min(len(test)-1, prediction_length-1)], color="gray", alpha=0.2, label="Forecast Region")
plt.title("Electricity Usage Forecast (Chronos-T5 via ChronosPipeline)")
plt.xlabel("Date")
plt.ylabel("Avg Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np

# Convert forecast to NumPy array and average across multiple predictions
forecast_array = np.array(forecast_values)  # shape: [num_series, prediction_length]
mean_forecast = forecast_array.mean(axis=0)  # shape: [prediction_length]

# Align to test length
true_values = test.values
mean_forecast = mean_forecast[:len(true_values)]

# Evaluate MAPE
mape = mean_absolute_percentage_error(true_values, mean_forecast)
print("MAPE:", mape)

import matplotlib.pyplot as plt

# Plot actual usage and averaged forecast
plt.figure(figsize=(12, 6))

# Plot full historical actual usage
plt.plot(avg_usage.index, avg_usage, label="Actual Usage", linewidth=2)

# Plot the forecasted values (averaged across all series)
plt.plot(test.index[:len(mean_forecast)], mean_forecast, label="Forecast (Mean)", linestyle="--", linewidth=2)

# Highlight the forecast period
plt.axvspan(test.index[0], test.index[min(len(mean_forecast)-1, len(test)-1)], color="gray", alpha=0.2, label="Forecast Period")

# Formatting
plt.title("Electricity Usage Forecast - Averaged Chronos Predictions")
plt.xlabel("Date")
plt.ylabel("Average Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error
from scipy.signal import savgol_filter

# 1. Load and preprocess dataset
df = pd.read_csv("cluster_3.csv")
df = df.drop(columns=["Unnamed: 0", "cluster"])
df = df.set_index("account").T
df.index = pd.to_datetime(df.index)
avg_usage = df.mean(axis=1)

# 2. Apply Savitzky-Golay smoothing (no clipping)
smoothed = savgol_filter(avg_usage.values, window_length=9, polyorder=2)
avg_usage = pd.Series(smoothed, index=avg_usage.index)

# 3. Define train/test split dates
train_end = '2014-10-30'
test_start = '2014-11-01'
test_end = '2014-12-31'

train = avg_usage[:train_end]
test = avg_usage[test_start:test_end]
true_values = test.values  # already smoothed

# 4. Prepare context window (last 60 days of training)
context_window = torch.tensor(train[-60:].values, dtype=torch.float32).unsqueeze(0)

# 5. Load Chronos-T5 model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="auto",
    torch_dtype=torch.float32
)

# 6. Predict for full test horizon (6 months = 184 days)
prediction_length = len(test)
forecast_tensor = pipeline.predict(context_window, prediction_length)
forecast_values = forecast_tensor[0].tolist()

# Flatten if nested
if isinstance(forecast_values[0], list):
    forecast_values = [item for sublist in forecast_values for item in sublist]
forecast_values = np.array(forecast_values[:len(true_values)])  # Align length

# 7. Compute MAPE
mape = mean_absolute_percentage_error(true_values, forecast_values)
print("MAPE:", mape)

# 8. Plot results
plt.figure(figsize=(20, 7))

# Plot training data
plt.plot(train.index, train.values, label="Training Data", color='blue')

# Create matching index for forecast/test period
forecast_index = test.index[:len(forecast_values)]

# Plot true test values
plt.plot(forecast_index, true_values, label="True Test Data", color='green')

# Plot forecasted values
plt.plot(forecast_index, forecast_values, label="Forecasted Values", color='red', linestyle='--')

# Aesthetics
plt.title("Electricity Usage – Training, Test, and Forecast")
plt.xlabel("Date")
plt.ylabel("Average Electricity Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# 7. Plot: Training, True Test, and Forecast
plt.figure(figsize=(20, 7))

# Plot training data
plt.plot(train.index, train.values, label="Training Data", color='blue')

# Create matching index for forecast/test period
forecast_index = test.index[:len(forecast_values)]

# Plot true test values
plt.plot(forecast_index, true_values, label="True Test Data", color='green')

# Plot forecasted values
plt.plot(forecast_index, forecast_values, label="Forecasted Values", color='red', linestyle='--')

# Aesthetics
plt.title("Electricity Usage – Training, Test, and Forecast")
plt.xlabel("Date")
plt.ylabel("Average Electricity Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error

# 1. Load and process the dataset
df = pd.read_csv("cluster_0.csv")
df = df.drop(columns=["Unnamed: 0", "cluster"])
df = df.set_index("account").T
df.index = pd.to_datetime(df.index)
avg_usage = df.mean(axis=1)

# 2. Time splits
train_end = '2014-06-30'
test_start = '2014-07-01'
test_end = '2014-12-31'

train = avg_usage[:train_end]
test = avg_usage[test_start:test_end]
true_values = test.values  # Define true values here

# 3. Use last 60 days of training as context for forecasting
context_window = torch.tensor(train[-60:].values, dtype=torch.float32).unsqueeze(0)

# 4. Load Chronos-T5 model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="auto",
    torch_dtype=torch.float32
)

# 5. Predict for full test horizon (6 months = 184 days)
prediction_length = len(test)
forecast_tensor = pipeline.predict(context_window, prediction_length)
forecast_values = forecast_tensor[0].tolist()

# Flatten nested list if needed
if isinstance(forecast_values[0], list):
    forecast_values = [item for sublist in forecast_values for item in sublist]

forecast_values = np.array(forecast_values)

# Align lengths
forecast_values = forecast_values[:len(true_values)]

# 6. Compute MAPE
mape = mean_absolute_percentage_error(true_values, forecast_values)
print(" MAPE:", mape)

# 7. Plot: Training + Actual vs Predicted
plt.figure(figsize=(14, 6))

# Plot training data
plt.plot(train.index, train.values, label="Training Data", color='blue')

# Plot true test values
plt.plot(test.index, true_values, label="Actual Usage (Test)", color='black', linestyle='--')

# Plot predicted values in green
plt.plot(test.index[:len(forecast_values)], forecast_values, label="Forecast (Chronos-T5)", color='green')

plt.title("Electricity Usage Forecast – Chronos-T5 (6-Month Horizon)", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Average Electricity Usage", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# # Prophet


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from scipy.signal import savgol_filter
from prophet import Prophet

# 1. Load and preprocess dataset
df = pd.read_csv("cluster_0.csv")
df = df.drop(columns=["Unnamed: 0", "cluster"])
df = df.set_index("account").T
df.index = pd.to_datetime(df.index)
avg_usage = df.mean(axis=1)

# 2. Apply Savitzky-Golay smoothing (no clipping)
smoothed = savgol_filter(avg_usage.values, window_length=9, polyorder=2)
avg_usage = pd.Series(smoothed, index=avg_usage.index)

# 3. Define train/test split dates
train_end = '2014-10-30'
test_start = '2014-11-01'
test_end = '2014-12-31'

train = avg_usage[:train_end]
test = avg_usage[test_start:test_end]
true_values = test.values

# 4. Prepare data for Prophet
prophet_df = train.reset_index()
prophet_df.columns = ['ds', 'y']

# 5. Fit the Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(prophet_df)

# 6. Create future dataframe and forecast (add +1 to capture Dec 31)
future = model.make_future_dataframe(periods=len(test) + 1, freq='D')
forecast = model.predict(future)

# 7. Align forecast with test index
forecast = forecast.set_index('ds')
forecast = forecast.loc[test.index.intersection(forecast.index)]  # Safe intersection
forecast_values = forecast['yhat'].values

# 8. Compute MAPE
mape = mean_absolute_percentage_error(true_values, forecast_values)
print("MAPE (Prophet):", mape)

# 9. Plot results
plt.figure(figsize=(14, 6))

# Plot training data
plt.plot(train.index, train.values, label="Training Data", color='blue')

# Plot actual test values
plt.plot(test.index, true_values, label="Smoothed Actual Usage (Test)", color='black', linestyle='--')

# Plot forecast values
plt.plot(test.index[:len(forecast_values)], forecast_values, label="Forecast (Prophet)", color='green', linewidth=2)

# Labels and layout
plt.title("Electricity Usage Forecast – Prophet (Nov–Dec 2014)", fontsize=16, weight='bold', color='#006600')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Average Electricity Usage (kWh)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 1. Load and preprocess dataset
df = pd.read_csv("cluster_0.csv")
df = df.drop(columns=["Unnamed: 0", "cluster"])
df = df.set_index("account").T
df.index = pd.to_datetime(df.index)
original_avg_usage = df.mean(axis=1)

# 2. Apply Savitzky-Golay smoothing
smoothed = savgol_filter(original_avg_usage.values, window_length=9, polyorder=2)
avg_usage = pd.Series(smoothed, index=original_avg_usage.index)

# 3. Identify dips: where original is significantly lower than smoothed
residual = original_avg_usage - avg_usage
dip_threshold = np.percentile(residual, 5)  # Adjust threshold as needed
dip_indices = residual[residual < dip_threshold].index
dip_values = original_avg_usage[dip_indices]

# 4. Plot with elegant colors
plt.figure(figsize=(20, 8))
plt.plot(original_avg_usage.index, original_avg_usage.values,
         label="Original Data", color='steelblue', alpha=0.7, linewidth=2)
plt.plot(avg_usage.index, avg_usage.values,
         label="Smoothed Data (Savitzky-Golay)", color='darkorange', linewidth=2)

# Mark dips with crimson red circles with white edge
plt.scatter(dip_indices, dip_values, color='crimson',
            edgecolors='white', s=80, label='Smoothed-Out Dips', zorder=5)

# Beautify layout
plt.title("Original vs Savitzky-Golay Smoothed Electricity Usage with Dips Highlighted",
          fontsize=16, weight='bold', color='#333333')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Average Electricity Usage (kWh)", fontsize=12)
plt.legend(fontsize=12, frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from scipy.signal import savgol_filter
from prophet import Prophet

# Cluster CSV files
clusters = ["cluster_0.csv", "cluster_1.csv", "cluster_2.csv", "cluster_3.csv"]
mapes_per_cluster = []
n_repetitions = 10  # Prophet runs per cluster

for cluster in clusters:
    cluster_mapes = []

    for _ in range(n_repetitions):
        # 1. Load and preprocess data
        df = pd.read_csv(cluster)
        df = df.drop(columns=["Unnamed: 0", "cluster"])
        df = df.set_index("account").T
        df.index = pd.to_datetime(df.index)
        avg_usage = df.mean(axis=1)

        # Optional: Add slight noise to avoid deterministic output
        avg_usage += np.random.normal(0, 0.01, size=len(avg_usage))

        # 2. Smooth data
        smoothed = savgol_filter(avg_usage.values, window_length=9, polyorder=2)
        avg_usage = pd.Series(smoothed, index=avg_usage.index)

        # 3. Train-test split
        train_end = '2014-10-30'
        test_start = '2014-11-01'
        test_end = '2014-12-31'

        train = avg_usage[:train_end]
        test = avg_usage[test_start:test_end]
        true_values = test.values

        # 4. Prophet formatting
        prophet_df = train.reset_index()
        prophet_df.columns = ['ds', 'y']

        # 5. Train Prophet model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(test) + 1, freq='D')
        forecast = model.predict(future)

        # 6. Align forecast
        forecast = forecast.set_index('ds')
        forecast = forecast.loc[test.index.intersection(forecast.index)]
        forecast_values = forecast['yhat'].values[:len(true_values)]

        # 7. Compute MAPE (%)
        mape = mean_absolute_percentage_error(true_values, forecast_values) * 100
        cluster_mapes.append(mape)

    mapes_per_cluster.append(cluster_mapes)
    print(f"Cluster {cluster} MAPEs (%):", cluster_mapes)

# 8. Box plot
plt.figure(figsize=(10, 6))
plt.boxplot(mapes_per_cluster,
            labels=[f"Cluster {i}" for i in range(len(clusters))],
            showmeans=True)

# Scatter plot of individual points (jittered)
for i, mape_list in enumerate(mapes_per_cluster):
    x = np.random.normal(loc=i + 1, scale=0.05, size=len(mape_list))
    plt.plot(x, mape_list, 'o', alpha=0.5, color='blue')

plt.title("MAPE Across Clusters – Prophet", fontsize=14, weight='bold')
plt.xlabel("Clusters", fontsize=12)
plt.ylabel("MAPE (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()