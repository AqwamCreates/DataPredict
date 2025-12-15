# Economy Systems

## Disclaimer

* I recommend you to use in-game currencies instead of the real-world ones.

   * It is far more difficult to make players into spending more real-world money than the in-game ones.

   * Additionally, there are a lot of potential legal issues when price gouging using real-world money.

## Models

* [Creating Willingness-To-Pay Prediction Model](EconomySystems/CreatingWillingnessToPayPredictionModel.md)

    * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Base Price Search Model](EconomySystems/CreatingBasePriceSearchModel.md)

    * Extremely useful if one of the in-game currencies is hard to get (especially if can be affected by external means like exploits and trading), leading to lower spending of that currency. This would then lead to item that is to be purchased requiring that particular in-game currency to be lower. This is because due to lack of buyer's supply to satisfy seller's, or in this case, our ML model's demand for that currency. As a result this leads to the in-game currency be viewed as "extremely valuable" by the game's ecosystem.

    * Recommended to use this with the Dynamic Pricing Models.
    
    * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* Dynamic Pricing For In-Game Currencies (Use TD Actor-Critic + Diagonal Gaussian Policy + Tanh activation outputs to multiply with base cost and then add with the base cost to get new effective cost.)

* Virtual Currency Velocity (Use non-linear KalmanFilter variants due to data not being linear.)

* Transaction Prediction (Use Markov. Predict next purchase type / timing.)
