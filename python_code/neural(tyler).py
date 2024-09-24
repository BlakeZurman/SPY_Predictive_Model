import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from PCA import perform_pca


# Use the first two principal components as features=
X_pca = pca_data[:, :2]

# Add VIX data and previous momentum as additional features
X_final = np.concatenate((X_pca, mean_numbers[['VIX_Close', 'SPY_Close_1', 'SPY_Close_2', 'SPY_Close_3', 'SPY_Close_4', 'SPY_Close_5', 'SPY_Close_6', 'SPY_Close_7', 'Total_Volume_1', 'Total_Volume_2', 'Total_Volume_3', 'Total_Volume_4', 'Total_Volume_5', 'Total_Volume_6', 'Total_Volume_7']].values), axis=1)

# Target variable (next-day delta in SPY)
y = mean_numbers['Target'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Neural Network Model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_final.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))  
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluation
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Prediction
new_data_pca = pca.transform(new_data)[:, :2]  
new_data_final = np.concatenate((new_data_pca, new_data[['VIX_Close', 'SPY_Close_1', 'SPY_Close_2', 'SPY_Close_3', 'SPY_Close_4', 'SPY_Close_5', 'SPY_Close_6', 'SPY_Close_7', 'Total_Volume_1', 'Total_Volume_2', 'Total_Volume_3', 'Total_Volume_4', 'Total_Volume_5', 'Total_Volume_6', 'Total_Volume_7']].values), axis=1)

predictions = model.predict(new_data_final)
