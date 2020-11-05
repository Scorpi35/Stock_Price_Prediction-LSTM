import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
from datetime import date
from keras.models import load_model

# Extracting and Normalizing data
dataset_train = pd.read_csv('Datasets/Train_Data.csv')
training_set = dataset_train.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Defining Lists
X_test = []
Plot_Data = []

# Enter the date up to which stock price has to been predicted
Prediction_Date = input('Enter the date up to which stock prices have to be predicted in YYYY-MM-DD format:-')
year, month, day = map(int, Prediction_Date.split('-'))
Projected_Date = date(year, month, day)

# Getting Initial Date and Initial Index
df = pd.read_csv('Datasets/Initial_Data.csv')
Initial_Index = df.tail(1).index.item()
Initial_Index = Initial_Index + 1
Initial_Date = df.tail(1).iloc[0]['Date']
year, month, day = map(int, Initial_Date.split('-'))
Initial_Date = date(year, month, day)

# Total number of days in between two dates
delta = Projected_Date - Initial_Date
total_days = delta.days

# Generating tomorrow's day in comparison to recent predicted date
tomorrow = Initial_Date + datetime.timedelta(days=1)
tomorrow1 = deepcopy(str(tomorrow))

for i in range(Initial_Index, Initial_Index + total_days):

    X_test = []
    tomorrow1 = deepcopy(str(tomorrow))

    df = pd.read_csv('Datasets/Initial_Data.csv')
    inputs = df['Open'].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # Populating X_test with data
    X_test.append(inputs[i-60: i, 0])
    X_test1 = deepcopy(np.array(X_test))
    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))

    # Loading the trained model
    open_model = load_model("Trained_Model/Trained_Model.h5")

    # Predicting stock prices
    predicted_value = open_model.predict(X_test1)
    predicted_value = sc.inverse_transform(predicted_value)

    # Converting list to number
    b = predicted_value.ravel()
    b = float(b)

    # Inserting new row
    df.loc[i + 1] = [tomorrow1, b]
    df.to_csv('Datasets/Initial_Data.csv', index=False)

    print(tomorrow1 + " = " + str(b))

    # Populating Plot_Data with predited values
    Plot_Data.append([b])

    # Evaluating next date
    tomorrow = tomorrow + datetime.timedelta(days=1)


# Plotting Predicted Stock Prices
plt.plot(Plot_Data, color='green', label='Predicted Microsoft Stock Price')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Microsoft Stock Price')
plt.legend()
plt.show()




