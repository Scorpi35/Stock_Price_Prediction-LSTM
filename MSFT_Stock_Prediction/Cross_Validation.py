import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.models import load_model

# Opening LSTM model
open_name = "Trained_Model/Trained_Model.h5"
regressor = load_model(open_name)

# Making MinMaxScaler from training dataset
dataset_train = pd.read_csv('Datasets/Train_Data.csv')
training_set = dataset_train.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range=(0, 1))
sc.fit_transform(training_set)

# Cross-validation dataset
dataset_test = pd.read_csv('Datasets/Cross_Data.csv')
real_stock_price = dataset_test.iloc[0:277, 1:2].values
dates = dataset_test['Date'].values

# Making inputs
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Populating X_test and real with data
X_test = []
real = []
for i in range(60, 337):
    X_test.append(inputs[i-60:i, 0])
    real.append(inputs[i])

# Shaping X_test
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Shaping real
real = np.array(real)
real = sc.inverse_transform(real)

# Predicting the stock prices
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)

# Plotting Predicted Stock Prices and Real Stock Prices
plt.figure(figsize=(16, 8))
plt.plot(real_stock_price, color='black', label='Microsoft')
plt.plot(predicted_stock_price, color='green', label='Predicted Microsoft Stock Price')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Microsoft Stock Price')
plt.legend()
plt.show()

# Finding Out RMS
rms = np.sqrt(np.mean(np.power((real_stock_price-predicted_stock_price), 2)))
print("--------------------------------------------")
print("Root Mean Square Error:- " + str(rms))



