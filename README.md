# StockPrice with LSTM

## Project Description
Predicting stock price demands lots of input features. Instead of determing every possible input feature, we can analyze the pattern in which stock price is moving and predict the price. Addressing this very problem, we decided to use LSTM (a type of RNN) to anayze the pattern in which stock is regulating and predict future stock prices. Conclusively,it's a time series prediction.

## Dataset
* Dataset from https://www.macrotrends.net/stocks/charts/MSFT/microsoft/stock-price-history
* Training Samples:- 8018
* Test Samples:- 277

## Programming language and libraries
* Language:- Python
* Libraries:- sklearn, keras, numpy, matplot, pandas

## Algorithm
1. Data collection
2. Data cleansing
3. Training neural network with Sequential Model (LSTM)
4. Prediction of stock prices up to the required date

## Training Description
* Model:- LSTM (Sequential), 4 LSTM layer, 4 Dropout layer, 1 Dense layer
* Activation:- Sigmoid
* Optimizer:- Adam
* Epoch:- 150
* Batch Size:- 32

## Results

  - Accuracy (Cross-Validation):- 0.45 
  - Accuaracy (Tes):- 0.40

## Screenshots

###### User Input
![alt text](https://github.com/Scorpi35/StockPrice-LSTM/blob/master/Screenshots/User_Input.png)

###### Cross-Validation Visualization
![alt text](
https://github.com/Scorpi35/StockPrice-LSTM/blob/master/Screenshots/Cross-Validation%20Data%20Visualization.png)

###### Test Visualization
![alt text](https://github.com/Scorpi35/StockPrice-LSTM/blob/master/Screenshots/Test_Data_Visualization.png)

###### 7 days Prediction
![alt text](https://github.com/Scorpi35/StockPrice-LSTM/blob/master/Screenshots/7days_Prediction.png)


## How to run the program?

#### Installing required dependencies
* Run **pip install -r requirements.txt**

#### Running the program
* Run **python Prediction.py**
* Enter date upto which stock prices have to be predicted.





