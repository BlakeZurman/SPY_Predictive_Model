import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

dataspy = pd.read_csv('SPYdata.csv')
datavix = pd.read_csv('VIXdata.csv')

spy_features = dataspy[features1]  # Make sure to use the same features as used in training
vix_features = datavix[features2]

scalar1 = StandardScaler()
scalar2 = StandardScalar()



spy_scaled = scaler1.transform(spy_features)
vix_scaled = scaler2.transform(vix_features)

spy_pca = pca1.transform(spy_scaled)
vix_pca = pca2.transform(vix_scaled)

principal_components1 = pca1.fit_transform(data1_scaled)
principal_components2 = pca2.fit_transform(data2_scaled)

X = np.concatenate([spy_pca, vix_pca], axis=1)
y = dataspy['Target'].values 


model = build_model(X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')




predictions = model.predict(new_X)

# Optionally, convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Target'])

# Print or save the predictions
print(predictions_df)

# Predictions
# For new predictions, ensure you scale and PCA-transform the new data similarly
# new_data_scaled = scaler.transform(new_data)
# new_data_pca = pca.transform(new_data_scaled)
# predictions = model.predict(new_data_pca)

# Print or use the predictions as needed
