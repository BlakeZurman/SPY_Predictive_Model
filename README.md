**Mission Statement** 
Using SKlearn we are concerned with writing a predictive model that attempts to quote the daily closing prices for the $SPY index. We used the Alphavantage® API to receive the historical daily close price and daily volume of the $SPY Index, as well as the historical daily close price of the $VIX volatility index for fiscal year 2023. These were given as CSV’s to parse.

**Group contribution**
My group and I came up with the idea for this project because of a shared interest in finance. We got to work finding a data set on Kaggle and using numerous data science techniques to clean and prep the data set. In addition, we used PCA algorithms to eliminate some variables that had a low impact on the data (small eigen values).

**My Individual Contribution to the Project**
I used this project as an opportunity to show my knowledge of basic neural network methods. I was tasked with train an LSTM model that used the cleaned SPY dataset. 

**Implementation and Libraries**
We made usage of Principal Component Analysis method to allow for simplification of data points, as well as Long Short-term Memory Neural Network for prediction. Libraries include: Pandas Numpy Sklearn Keras Matplotlib

**Results of the Model**
Through testing, we found that using approximately 80% of the data set to train the model, and 20% to test it yielded relatively valuable results. 
**After 100 epochs** 
Train loss: 0.0027 Test Loss: 0.0032
Train Predictions: [395.95715] [396.85263] [396.50003] [396.26282] [396.87833] 
Test Predictions: [431.5172] [428.293  ] [426.2334] [424.0384] [421.2292]



![Image](https://github.com/user-attachments/assets/f9c27bc7-c251-45f9-ba0d-c02713854de3)


