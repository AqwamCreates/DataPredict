# Economy Systems

* Willingness-to-Pay Prediction (Use BayesianLinearRegression Or BayesianQuantileLinearRegression. Gives uncertainty estimates.)

* Dynamic Pricing For In-Game Currencies (Use TD Actor-Critic + Diagonal Gaussian Policy + Tanh activation outputs to multiply with base cost and then add with the base cost to get new effective cost.)

* Virtual Currency Velocity (Use non-linear KalmanFilter variants due to data not being linear.)

* Transaction Prediction (Use Markov. Predict next purchase type / timing.)
