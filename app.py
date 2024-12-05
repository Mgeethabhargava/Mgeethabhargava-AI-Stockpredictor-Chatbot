from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from dataprocess import load_saved_model, predict_with_loaded_model, fetch_stock_data, preprocess_data, prepare_dataset,prepare_prediction_data, forecast_future_prices, multimodel, predict_intent,generate_graph

app = Flask(__name__)
# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Load the pre-trained AI model
MODEL_PATH = os.path.join(current_dir, 'AI_stock_model.h5')
model_save_path = os.path.join(current_dir, 'multi_intent_model.pt')
model = load_saved_model(MODEL_PATH)
intent_model, tokenizer, mlb, device = multimodel(model_save_path)
# Create a directory for saving graph images
GRAPH_FOLDER = os.path.join('static', 'graphs')
os.makedirs(GRAPH_FOLDER, exist_ok=True)
API_KEY = 'FKNN7J13JVGS9NHY'
# API_KEY = 'JO9JU1XC70184W0G'
# API_KEY = '57EMD3ZLNVRDQJTJ'

@app.route('/')
def home():
    messages = { "sender": 'bot', "text": "Hi! I'm FinanceBot, your Query ChatBot ❤️" }
    return render_template('index.html', messages=messages)

@app.route('/get', methods=['GET'])
def chatbot_response():
    user_message = request.args.get('msg')
    predicted_intents = predict_intent(user_message, intent_model, tokenizer, device, mlb)
    print(f"Input: {user_message}")
    if not predicted_intents:
        print("Failed case")
        return jsonify({"bot_reply": "I didn't get that. Could you clarify?"})
    
    # Simple logic for chatbot replies
    if "greet" == predicted_intents.lower():
         return jsonify({"bot_reply": f"Hello"}) 
    elif 'aapl'== predicted_intents.lower():
         # Parameters
         symbol = 'AAPL'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'         
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'msft'== predicted_intents.lower():
         # Parameters
         symbol = 'MSFT'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'         
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'googl'== predicted_intents.lower():
         # Parameters
         symbol = 'GOOGL'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'  
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'amzn'== predicted_intents.lower():
         # Parameters
         symbol = 'AMZN'
         print(symbol)
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'  
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'bp.l'== predicted_intents.lower():
         # Parameters
         symbol = 'BP.L'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'  
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'hsba.l'== predicted_intents.lower():
         # Parameters
         symbol = 'HSBA.L'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'           
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'tsco.l'== predicted_intents.lower():
         # Parameters
         symbol = 'TSCO.L'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'  
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif 'jd.l'== predicted_intents.lower():
         # Parameters
         symbol = 'JD.L'
         lookback_period = 30
         # Preprocess the data for the specific stock
         raw_data = fetch_stock_data(API_KEY, [symbol])
         stock_data = preprocess_data(raw_data[symbol])
         X, y, scaler = prepare_dataset(stock_data, lookback=lookback_period)         
         # Prepare the most recent data for prediction
         prediction_data = prepare_prediction_data(stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler)
         # Predict the next day's price
         predicted_price = predict_with_loaded_model(model, prediction_data, scaler)
         future_prices = forecast_future_prices(model, stock_data[['Close', 'SMA_20', 'SMA_50', 'RSI']].values, scaler, n_days=10)
         graph_filename = generate_graph(stock_data, 10, future_prices, symbol, GRAPH_FOLDER)
         graph_url = f'graphs/{graph_filename}'  
         #print(f"The predicted next-day price for {symbol} is: ${predicted_price:.2f}")
         return jsonify({"bot_reply": f"The predicted next-day price for {symbol} is: {predicted_price} and future_prices are {future_prices}", 'graph_url': graph_url})
    elif "sendoff" == predicted_intents.lower():
         return jsonify({"bot_reply": f"Bye Meet You again"})
    else:
        return jsonify({"bot_reply": "I'm here to help with financial queries. Can you provide more details?"})

if __name__ == '__main__':
    app.run(debug=True)
