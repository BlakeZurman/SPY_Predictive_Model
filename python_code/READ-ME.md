# Mission Statement

Using SKlearn, we intend to have a predictive model for the S&P500's next-day price. We have daily close price, total daily volume of SPY for all of 2023, as well as daily close price of VIX for all of 2023. We intend to take a set 7 days of the SPY, to try to predict the next day's closing price. We will do this by using the price-change/volume of SPY in that 7 days, as well as the price of the VIX to make a decision.

Charlie/Blake's idea:
Select a handful of equities in SPY that are positive, and then attempt to predict SPY with these equities

Alternate:
Make a correlation between price of the VIX and the next-day price of SPY.

We use a simple regression model with a neural network.


3/3/24 addition: PCA file is made, we will be using the principal components as input features for the NN.

