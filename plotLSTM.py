import pandas as pd
#data manipulation
import numpy as np
#numerical computing
from sklearn.model_selection import train_test_split
#splits the data sets into training and testing
from sklearn.preprocessing import MinMaxScaler
#used for feature scaling
from keras.models import Sequential
#constructs the nueral network layer by layer
from keras.layers import LSTM, Dense
#long short term memory layer for recurrent neural network architecture
#dense layer for fully connected layers
import matplotlib.pyplot as plt
#plotting graphs

#READING DATA
data = pd.read_csv('SPYdata.csv')
#reads the data inside of the CSV file (uses pandas)
closing_prices = data['Close'].values
#extracts the closing prices from the dataset

#DATA PREPROCESSING
####neural nets typically expect data in 2-D arrays
scaler = MinMaxScaler(feature_range=(0, 1))
#initializes MinMaxScaler to scale the closing price between 0 and 1 
#normalizing enusres the same scale: imperative for distance measures or gradients
closing_prices_scaled = scaler.fit_transform(closing_prices.reshape(-1, 1))
#shapes the closing prices using the MinMaxScaler
#changes the 1-D array from '.values' to a 2-D array using 'reshape(-1, 1)'
#makes closing prices a seperate sample, allowing it to sacle each price individually
window_size = 10 
#defines the window size for the input sequence

#CREATING SEQUENCES FOR LSTM
X, y = [], []
#initializes empty set lists for features and labels
for i in range(len(closing_prices_scaled) - window_size):
    #loop through the scaled closing prices to create input sequences and labels
    X.append(closing_prices_scaled[i:i+window_size])
    #each element 'X' is a sequence of 'window_size' consecutive closing prices
    y.append(closing_prices_scaled[i+window_size])
    #each element 'y' is the next closing price after the corresponding sequence in 'X'
X, y = np.array(X), np.array(y)
#this loop essentially creates the input-output pairs for training the LSTM model

#SPLITING DATA INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#here 80% is used for training and 20% is used for testing

#BUILDING THE LSTM MODEL
model = Sequential()
#initializes a sequential model
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#adds the first LSTM layer with 50 units, specifying the input shape
model.add(LSTM(units=50))
#adds a second LSTM layer with 50 units
model.add(Dense(units=1))
#adds a dense layer with 1 unit for output
model.compile(optimizer='adam', loss='mean_squared_error')
#compiles the model with Adam (adaptive moment estimation) optimizer and mean squared error loss function
#adam dynamically adjusts the learning rate during training for each parameter based on the first and second moments of the gradients
#adam incorporates momentum, which accelerates convergence by accumulating exponentially weighted moving averages of past gradients.
#adam corrects biases in first and second moments of the gradients (with emphasis of the early training iterations because the initialized valeus are random)
#adam uses L2 regularization (weight decay) to penalize large weights to prevent over fitting

####LSTM layers use the tanh activation function for the reccurent step and the sigmoid function for the gates

#TRAINING THE MODEL
model.fit(X_train, y_train, epochs=100, batch_size=32)
#trains the model for 100 epochs with a batch size of 32
#epochs initializes, does a training loop, forward passes, backprogagates, and updates parameters
#the number of epochs depends on the data set and converenge behavior of the model during training

#EVALUATING THE MODEL
train_loss = model.evaluate(X_train, y_train, verbose=0)
#evaluates the models loss on the training data
test_loss = model.evaluate(X_test, y_test, verbose=0)
#evaluates the models loss on the test data
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

#MAKING PREDICTIONS
train_predictions = model.predict(X_train)
#predictions on the training data
test_predictions = model.predict(X_test)
#predictions on the test data

train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
#inverse scales the predicted values to get the actual stock prices 
#basically reverses the MinMaxScalar
#scaled predictions are proportional to scaled training data
print("Train Predictions:", train_predictions[:5])
print("Test Predictions:", test_predictions[:5])

#PLOTTING PREDICTIONS
plt.figure(figsize=(10, 6))
plt.plot(np.arange(window_size, len(train_predictions) + window_size), train_predictions, label='Train Predictions')
plt.plot(np.arange(len(train_predictions) + window_size, len(train_predictions) + len(test_predictions) + window_size), test_predictions, label='Test Predictions')
plt.plot(closing_prices, label='Actual Closing Prices')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('SPY Stock Price Prediction')
plt.legend()
plt.show()
