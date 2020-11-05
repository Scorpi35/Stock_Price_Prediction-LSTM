import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pickle

# Making pandas data frame for Stock_Data.csv
dataset_train = pd.read_csv('Datasets/Train_Data.csv')
training_set = dataset_train.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating X_train and y_train
X_train = []
y_train = []

# Populating X_train and y_train with values
for i in range(60, 8018):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Starting sequential regression neural network
# 5 layers total
regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, activation='sigmoid', input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True, activation='sigmoid'))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True, activation='sigmoid'))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training LSTM model and finding best fit curve
regressor.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=32, verbose=1)

regressor.save("predict_open_model.h5")

# Saving LSTM model
filename = 'Trained_Model/Predict_Open_Model1.sav'
pickle.dump(regressor, open(filename, "wb"))


