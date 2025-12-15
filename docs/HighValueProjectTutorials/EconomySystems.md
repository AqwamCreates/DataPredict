# Economy Systems

## Disclaimer

* Recommend to use in-game currencies and not the real-world currencies. It is far more difficult to make players into spending more real money than the in-game ones. Additionally, there is a potential legal issues when price gouging using real money.

## Models

* Willingness-to-Pay Prediction (Use BayesianLinearRegression or BayesianQuantileLinearRegression. Gives uncertainty estimates.)

* Dynamic Pricing For In-Game Currencies (Use TD Actor-Critic + Diagonal Gaussian Policy + Tanh activation outputs to multiply with base cost and then add with the base cost to get new effective cost.)

* Base Price Search (Use EM, Fuzzy C-Means or K-Means. Extract model parameters to get the base price. Recommended to use this with the Dynamic Pricing Models)

    * Extremely useful if one of the in-game currencies is hard to get (especially if can be affected by external means like exploits and trading), leading to lower spending of that currency. This would then lead to item that is to be purchased requiring that particular in-game currency to be lower. This is because due to lack of buyer's supply to satisfy seller's, or in this case, our ML model's demand for that currency. As a result this leads to the in-game currency be viewed as "extremely valuable" by the game's ecosystem.

* Virtual Currency Velocity (Use non-linear KalmanFilter variants due to data not being linear.)

* Transaction Prediction (Use Markov. Predict next purchase type / timing.)
