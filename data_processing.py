import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import streamlit as st


import math
#from pandas_datareader import data as pdr
from datetime import datetime,date
import yfinance as yf
#yf.pdr_override()
import joblib
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title, normalize):

    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    else:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
	
def calculate_stochastic(df, period=14):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    return k

def calculate_williams_r(df, period=14):
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    return -100 * ((high_max - df['Close']) / (high_max - low_min))

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
    
def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2
    
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()
    
def calculate_bollinger_bands(prices, period=20, std_dev=2):
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return upper_band, lower_band

def generate_features(df):
    df['Adj Open'] = df.apply(lambda row: row['Open'] * (row['Adj Close'] / row['Close']), axis=1)
    df['Adj High'] = df.apply(lambda row: row['High'] * (row['Adj Close'] / row['Close']), axis=1)
    df['Adj Low'] = df.apply(lambda row: row['Low'] * (row['Adj Close'] / row['Close']), axis=1)
    df['Adj Vol'] = df.apply(lambda row: row['Volume'] * (row['Close'] / row['Adj Close']), axis=1)
    #df['Adj Value'] = df.apply(lambda row: row['Adj Close'] * row['Adj Vol'], axis=1)
    
    
    df['Diff'] = df['Adj Close'].pct_change()
    # df['Diff_Vol'] = df['Adj Vol'].pct_change()
    df['EMA_7'] = df['Adj Close'].ewm(span=7, adjust=False).mean()
    df['EMA_14'] = df['Adj Close'].ewm(span=14, adjust=False).mean()
    df['EMA_21'] = df['Adj Close'].ewm(span=21, adjust=False).mean()

    df['MA_7_volume'] = df['Adj Vol'].rolling(window=7).mean()
    df['MA_14_volume'] = df['Adj Vol'].rolling(window=14).mean()
    df['MA_21_volume'] = df['Adj Vol'].rolling(window=21).mean()

    # Calculate new features
    df['Daily_Close'] = df.apply(lambda row: (row['Adj Close'] - row['Adj Close']).shift(1) / row['Adj Close'].shift(1), axis=1)
    df['Daily_Volume'] = df.apply(lambda row: (row['Adj Vol'] - row['Adj Vol']).shift(1) / row['Adj Vol'].shift(1), axis=1)

    df['Close_to_Open'] = df.apply(lambda row: (row['Adj Close'] - row['Adj Open']) / row['Adj Close'], axis=1)
    df['Close_to_High'] = df.apply(lambda row: (row['Adj Close'] - row['Adj High']) / row['Adj Close'], axis=1)
    df['Close_to_Low'] = df.apply(lambda row: (row['Adj Close'] - row['Adj Low']) / row['Adj Close'], axis=1)

    df['Volume_Change_7'] = df.apply(lambda row: (row['Adj Vol'] - row['MA_7_volume']) / row['MA_7_volume'], axis=1)
    df['Volume_Change_14'] = df.apply(lambda row: (row['Adj Vol'] - row['MA_14_volume']) / row['MA_14_volume'], axis=1)
    df['Volume_Change_21'] = df.apply(lambda row: (row['Adj Vol'] - row['MA_21_volume']) / row['MA_21_volume'], axis=1)
    df['EMA_7_Change'] = df.apply(lambda row: (row['Adj Close'] - row['EMA_7']) / row['Adj Close'], axis=1)
    df['EMA_14_Change'] = df.apply(lambda row: (row['Adj Close'] - row['EMA_14']) / row['Adj Close'], axis=1)
    df['EMA_21_Change'] = df.apply(lambda row: (row['Adj Close'] - row['EMA_21']) / row['Adj Close'], axis=1)
    
    df['RSI_14'] = calculate_rsi(df['Adj Close'], period=14)
    df['MACD'] = calculate_macd(df['Adj Close'])
    df['ATR_14'] = calculate_atr(df, period=14)
    df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df['Adj Close'])
    df['Stochastic_K'] = calculate_stochastic(df)
    df['Williams_R'] = calculate_williams_r(df)

	
     # Select relevant features
    features = ['Adj Open','Adj High','Adj Low' ,'Adj Close', 'Adj Vol', 'Diff',
              'EMA_7', 'EMA_14', 'EMA_21', 'MA_7_volume','MA_14_volume' ,'MA_21_volume',
              'Daily_Close', 'Daily_Volume', 'Close_to_Open', 'Close_to_High', 'Close_to_Low',
    'Volume_Change_7', 'Volume_Change_14', 'Volume_Change_21',
    'EMA_7_Change', 'EMA_14_Change', 'EMA_21_Change',
    'RSI_14','MACD','ATR_14','Upper_BB','Lower_BB','Stochastic_K','Williams_R'#,'Adj_Value'
	       ]

  # Add last close price and volume of the last 21 days
    for i in range(1, 7):

      df[f'Open_{i}'] = df['Open'].shift(i)
      df[f'High_{i}'] = df['High'].shift(i)
      df[f'Low_{i}'] = df['Low'].shift(i)
      df[f'Close_{i}'] = df['Adj Close'].shift(i)
      df[f'Volume_{i}'] = df['Volume'].shift(i)
      df[f'Diff_{i}'] = df['Diff'].shift(i)
      #df[f'Diff_Vol_{i}'] = df['Diff_Vol'].shift(i)
      #df[f'Adj_Value_{i}'] = df['Adj_Value'].shift(i)
      df[f'EMA_7_{i}'] = df['EMA_7'].shift(i)
      df[f'EMA_14_{i}'] = df['EMA_14'].shift(i)
      df[f'EMA_21_{i}'] = df['EMA_21'].shift(i)
      df[f'MA_7_volume_{i}'] = df['MA_7_volume'].shift(i)
      df[f'MA_14_volume_{i}'] = df['MA_14_volume'].shift(i)
      df[f'MA_21_volume_{i}'] = df['MA_21_volume'].shift(i)

      df[f'Daily_Close_{i}'] = df['Daily_Close'].shift(i)
      df[f'Daily_Volume_{i}'] = df['Daily_Volume'].shift(i)
      df[f'Close_to_Open_{i}'] = df['Close_to_Open'].shift(i)
      df[f'Close_to_High_{i}'] = df['Close_to_High'].shift(i)
      df[f'Close_to_Low_{i}'] = df['Close_to_Low'].shift(i)
      df[f'Volume_Change_7_{i}'] = df['Volume_Change_7'].shift(i)
      df[f'Volume_Change_14_{i}'] = df['Volume_Change_14'].shift(i)
      df[f'Volume_Change_21_{i}'] = df['Volume_Change_21'].shift(i)
      df[f'EMA_7_Change_{i}'] = df['EMA_7_Change'].shift(i)
      df[f'EMA_14_Change_{i}'] = df['EMA_14_Change'].shift(i)
      df[f'EMA_21_Change_{i}'] = df['EMA_21_Change'].shift(i)


      features.extend([f'Open_{i}',f'High_{i}',f'Low_{i}',f'Close_{i}', f'Volume_{i}',#f'Diff_{i}',
                       #f'EMA_7_{i}',f'EMA_14_{i}',f'EMA_21_{i}',
                       # f'MA_7_volume_{i}',f'MA_14_volume_{i}',f'MA_21_volume_{i}',
                      #f'Daily_Close_{i}', f'Daily_Volume_{i}',
                      f'Close_to_Open_{i}',f'Close_to_High_{i}',f'Close_to_Low_{i}',
                      #f'Volume_Change_7_{i}',
                      # f'Volume_Change_14_{i}',f'Volume_Change_21_{i}',
                       #f'EMA_7_Change_{i}',
                       #f'EMA_14_Change_{i}',
                       #f'EMA_21_Change_{i}'
                       #f'Adj_Value_{i}'

                       ])
    return df,features
	
def generate_target_classification(df, target_days, target_percent):
    target_close = df['Adj Close'].shift(-target_days)
    df['Target'] = np.where(target_close > df['Adj Close'] * (1 + (target_percent/100)), 1, 0)
    df['Target_Close'] = target_close
    return df

    
def generate_target_regression(df, target_days):

    target_close = df['Adj Close'].shift(-target_days)
    df['Target'] = target_close
    return df
    
def process_split_data(symbol,target_days, target_percent,model_type):

  today = datetime.today().strftime('%Y-%m-%d')
  symbol = symbol.upper()
  df = yf.download(symbol + '.NS',"2018-01-01", today)
  
  
  df_features, features = generate_features(df)
  df_features = df_features.dropna()

  if model_type == 'Classification':
    df_features = generate_target_classification(df_features,target_days, target_percent)
  else:
    df_features = generate_target_regression(df_features, target_days)
  #df_features = df_features.dropna()
  X = df_features[features]
  
  # Select the target variable
  y = df_features['Target']
  #y_close = df_features['Target_Close']

  # Calculate number of validation days
  validation_days = 21
  # Separate validation data
  x_val = X[-validation_days:]
  y_val = y[-validation_days:]

  if model_type == 'Classification':
    y_close_val = df_features['Target_Close'][-validation_days:]
    merged_df = pd.concat([x_val[['Adj Close']], y_val], axis=1)
    merged_df.columns = ['Adj Close', 'Target']
    merged_df['Target_Close'] = y_close_val
    merged_df['Symbol'] = symbol
    merged_df['Diff'] = (merged_df['Target_Close'] - merged_df['Adj Close']) / merged_df['Adj Close'] * 100
    merged_df['New Target'] = np.where(merged_df['Target_Close'] > merged_df['Adj Close'] * (1 + (target_percent / 100)), 1, 0)
    merged_df['Target'] = merged_df['New Target']
    merged_df = merged_df[['Symbol', 'Adj Close', 'Target_Close', 'Target']]
  else:
    y_close_val = df_features['Target'][-validation_days:]
    merged_df = pd.concat([x_val[['Adj Close']], y_val], axis=1)
    merged_df.columns = ['Adj Close', 'Target']
    merged_df['Symbol'] = symbol
    merged_df = merged_df[['Symbol', 'Adj Close', 'Target']]
        


  # Remove validation data from X and y
  X = X[:-validation_days]
  y = y[:-validation_days]
  # Ensure that y corresponds to the same samples as X
  y = y.head(len(X))




  train_until= dt.datetime(year=2024, month=1, day=1)
  # Split the data based on train_until
  train_mask = X.index <= train_until
  X_train = X[train_mask]
  y_train = y[train_mask].dropna()
  X_test = X[~train_mask]
  y_test = y[~train_mask].dropna()

    # Ensure that y corresponds to the same samples as X
  #y_train = y_train.head(len(X_train))
  #y_test = y_test.head(len(X_test))
  # Split the data into training and testing sets
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  scaler = StandardScaler()

  scaled_X_train = scaler.fit_transform(X_train)
  scaled_X_test = scaler.transform(X_test)
  scaled_X_val = scaler.transform(x_val)
  
  return scaled_X_train, y_train, scaled_X_test, y_test, scaled_X_val, merged_df, scaler

def train_evaluate_classification_model(symbol,scaled_X_train, y_train, scaled_X_test, y_test, scaler, model_engine='LogisticRegression',plot_confusion=False,save_model=False):
    with st.spinner('Training model...'):
        # Initialize the model based on the chosen engine
        if model_engine == 'LogisticRegression':
            model = LogisticRegression(class_weight='balanced')
        elif model_engine == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight='balanced')
        elif model_engine == 'SVC':
            model = SVC(kernel='linear', probability=True, class_weight='balanced')
        elif model_engine == 'GradientBoostingClassifier':
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        elif model_engine == 'XGBoostClassifier':
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        else:
            raise ValueError("Invalid model engine specified")

        # Train the model
        model.fit(scaled_X_train, y_train)

          
        # Make predictions on the test set
        y_train_pred = model.predict(scaled_X_train)
        y_test_pred = model.predict(scaled_X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_test_pred)

        # Additional evaluation metrics (precision, recall, F1, ROC AUC)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_pred)

        if plot_confusion:
            plot_confusion_matrix(y_test, y_test_pred, title='Testing Data', normalize=False)
            plot_confusion_matrix(y_test, y_test_pred, title='Testing Data - Normalized', normalize=True)

        if save_model:
            import os
            os.makedirs("models", exist_ok=True)  # Create the "models" folder if it doesn't exist
            joblib.dump(model, os.path.join("models", f"{symbol}_model.h5"))

        return model, symbol,round(accuracy,2),round(precision,2),round(recall,2),round(roc_auc,2),round(f1,2)

def train_evaluate_regression_model(symbol, scaled_X_train, y_train, scaled_X_test, y_test, scaler, model_engine='LinearRegression', plot_residuals=False, save_model=False):
  """
  Trains and evaluates a regression model, returning performance metrics.

  Args:
      symbol (str): Stock symbol.
      scaled_X_train (array): Scaled training features.
      y_train (array): Training target values.
      scaled_X_test (array): Scaled testing features.
      y_test (array): Testing target values.
      scaler (object): Feature scaler used during training.
      model_engine (str, optional): Regression model engine (default: 'LinearRegression').
      plot_residuals (bool, optional): Whether to plot prediction residuals (default: False).
      save_model (bool, optional): Whether to save the trained model (default: False).

  Returns:
      tuple: (model, symbol, RMSE, R-squared)
  """

  with st.spinner('Training model...'):
    # Initialize the model based on the chosen engine
    if model_engine == 'LinearRegression':
      model = LinearRegression()
    elif model_engine == 'RandomForestRegressor':
      model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    elif model_engine == 'SVR':
      model = SVR()  # Adjust kernel and other parameters as needed
    elif model_engine == 'GradientBoostingRegressor':
      model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_engine == 'XGBRegressor':
      model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    else:
      raise ValueError("Invalid model engine specified")

    # Train the model
    model.fit(scaled_X_train, y_train)

    # Make predictions on the test set
    y_train_pred = model.predict(scaled_X_train)
    y_test_pred = model.predict(scaled_X_test)

    # Calculate regression metrics
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_test_pred)  # R-squared
    
#    percentage_changes = 100*np.abs(y_test/y_test_pred - 1)
#    mean_error =np.mean(percentage_changes)
    
    mae = mean_absolute_error(y_test, y_test_pred)  # Mean Absolute Error (added)
    mape = np.mean(np.abs(y_test - y_test_pred) / y_test) * 100  # Mean Absolute Percentage Error (added)


    # Optional: Plot prediction residuals
    if plot_residuals:
      import matplotlib.pyplot as plt
      plt.figure(figsize=(8, 6))
      plt.scatter(y_test, y_test_pred - y_test, s=3)
      plt.xlabel('True Target Value')
      plt.ylabel('Prediction Residual')
      plt.title('Prediction Residuals for {}'.format(symbol))
      plt.grid(True)
      st.pyplot()

    # Optional: Save the model
    if save_model:
      import os
      os.makedirs("models", exist_ok=True)  # Create the "models" folder if it doesn't exist
      joblib.dump(model, os.path.join("models", f"{symbol}_regression_model.pkl"))

    return model, symbol, round(rmse, 4), round(r2, 4),round(mae,4) ,round(mape,4)
