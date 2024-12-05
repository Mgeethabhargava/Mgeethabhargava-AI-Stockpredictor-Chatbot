import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries
from tensorflow.keras.models import load_model
import torch
import uuid
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "data.csv")

def load_saved_model(model_path):
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
    return model

def predict_with_loaded_model(model, prediction_data, scaler):
    scaled_prediction = model.predict(prediction_data)[0][0]
    rescaled_prediction = scaler.inverse_transform([[scaled_prediction, 0, 0, 0]])[0][0]
    return rescaled_prediction

def fetch_stock_data(api_key, symbols):
    ts = TimeSeries(key=api_key, output_format='pandas')
    stock_data = {}
    for symbol in symbols:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data.reset_index(inplace=True)
        data.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'
        }, inplace=True)
        data['Date'] = pd.to_datetime(data['date'])
        data.drop('date', axis=1, inplace=True)
        stock_data[symbol] = data.sort_values('Date')
    return stock_data

def preprocess_data(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().add(1).cumprod()))
    data.dropna(inplace=True)  # Drop rows with NaN values (from rolling)
    return data

def prepare_dataset(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_20', 'SMA_50', 'RSI']])
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # Predict the 'Close' price
    return np.array(X), np.array(y), scaler

def prepare_prediction_data(data, scaler, lookback=60):
    recent_data = data[-lookback:]  # Last `lookback` days of data
    scaled_recent_data = scaler.transform(recent_data)
    return np.expand_dims(scaled_recent_data, axis=0)  # Add batch dimension

def forecast_future_prices(model, recent_data, scaler, n_days=30):
    forecast = []
    input_sequence = recent_data[-60:]  # Start with the last 60 days of data
    for _ in range(n_days):
        input_sequence_scaled = scaler.transform(input_sequence)
        input_sequence_scaled = np.expand_dims(input_sequence_scaled, axis=0)
        predicted_price = model.predict(input_sequence_scaled)[0][0]
        forecast.append(predicted_price)
        next_row = np.append(input_sequence[-1][1:], predicted_price)
        input_sequence = np.vstack((input_sequence[1:], next_row))

    # Rescale forecast to original values
    forecast_rescaled = scaler.inverse_transform(
        np.hstack((np.array(forecast).reshape(-1, 1), np.zeros((n_days, recent_data.shape[1] - 1))))
    )[:, 0]
    return forecast_rescaled

def multimodel(model_save_path):
    data = pd.read_csv(csv_path)
    data['labels'] = data['labels'].str.split(',')
    mlb = MultiLabelBinarizer()
    data['labels'] = mlb.fit_transform(data['labels']).tolist()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(mlb.classes_), problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(model_save_path))
    intent_model = model.to(device)
    print(f"Model loaded from {model_save_path}")
    return intent_model, tokenizer, mlb, device

# Function to make predictions
def predict_intent(text, model, tokenizer, device, mlb, threshold=0.5):
    model.eval()
    with torch.no_grad():
        # Tokenize the input text
        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Get model predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Convert probabilities to labels
        predicted_labels = [mlb.classes_[i] for i, p in enumerate(probs) if p > threshold]
        return "".join(predicted_labels)

def generate_graph(stock_data, forecast_days, future_prices, symbol,graph_folder):
  # Plot the forecast
  plt.figure(figsize=(12, 6))
  plt.plot(range(len(stock_data)), stock_data['Close'].values, label='Historical Prices')
  plt.plot(range(len(stock_data), len(stock_data) + forecast_days), future_prices, label='Forecasted Prices')
  plt.title(f'{symbol} - Price Forecast')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.legend()
  # Save the graph with a unique name
  graph_filename = f'{uuid.uuid4()}.png'
  graph_path = os.path.join(graph_folder, graph_filename)
  plt.savefig(graph_path)
  plt.close()
  return graph_filename