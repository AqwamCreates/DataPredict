# Economy Systems

## Disclaimer

* Recommend to use in-game currencies and not the real-world currencies. It is far more difficult to make players into spending more real money than the in-game ones.

## Models

* Willingness-to-Pay Prediction (Use BayesianLinearRegression or BayesianQuantileLinearRegression. Gives uncertainty estimates.)

* Dynamic Pricing For In-Game Currencies (Use TD Actor-Critic + Diagonal Gaussian Policy + Tanh activation outputs to multiply with base cost and then add with the base cost to get new effective cost.)

* Base Price Search (Use EM, Fuzzy C-Means or K-Means. Extract model parameters to get the base price. Recommended to use this with the Dynamic Pricing Models)

* Virtual Currency Velocity (Use non-linear KalmanFilter variants due to data not being linear.)

* Transaction Prediction (Use Markov. Predict next purchase type / timing.)
