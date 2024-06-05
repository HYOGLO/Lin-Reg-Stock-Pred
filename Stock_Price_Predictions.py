import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Fetch stock data from Yahoo Finance
stock_data = yf.download('AAPL', start='2020-01-01', end='2022-01-01')

# Feature engineering
stock_data['Day'] = stock_data.index.day
stock_data['Month'] = stock_data.index.month
stock_data['Year'] = stock_data.index.year

# Define features and target variable
X = stock_data[['Day', 'Month', 'Year']]
y = stock_data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


'''
This code fetches stock data for Apple (AAPL) from Yahoo Finance,
engineers some basic features like day, month, and year,
uses a linear regression model to predict the closing prices
then evaluates the model using Mean Squared Error.


ChatGPT
To predict closing prices using mean squared error, build a predictive model,
evaluate its performance by calculating the mean squared error between predicted and actual closing prices,
and refine the model iteratively to minimize this error.
'''
