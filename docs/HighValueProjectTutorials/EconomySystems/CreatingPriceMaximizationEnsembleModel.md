# [Economy Systems](../EconomySystems.md) - Creating Price Maximization Ensemble Model

Hello guys! Today, I will be showing you on how to create a a pricing-based model that could predict the likelihood of a player would purchase an item for a particular price.

Currently, you need these to produce the model:

* A quantile regression model

* An ordinal regression model

* A player data that is stored in matrix

## Setting Up

Before we train our models, we will first need to construct the two-layer ensemble model.

### Defining Our Quantiles

The quantiles here describes the price gap between different users. Note that quantiles represent price thresholds, not probabilities.

```lua

local QuantilesList = {0.25, 0.5, 0.75, 0.90}

-- QuantilesList[1] = 25th percentile (conservative) price.
-- QuantilesList[2] = 50th percentile (balanced) price.
-- QuantilesList[3] = 75th percentile (aggressive) price.
-- QuantilesList[4] = 90th percentile (whale-focused) price.

```

### Quantile Regression Model Construction

In here, we have two options for selecting our quantile regression model.

| Model                               | Advantages                                                     | Disadvantages                                       |
|-------------------------------------|----------------------------------------------------------------|-----------------------------------------------------|
| Bayesian Quantile Linear Regression | Data efficient and is computationally fast for small datasets. | Assumes linear relationship and sensitive to noise. |
| Quantile Regression                 | Good against noisy data.                                       | Requires a lot of data.                             |

```lua

local DataPredict = require(DataPredict)

 -- QuantilesList is required for Quantile Regression model, but not for Bayesian Quantile Linear Regression model.

local QuantileRegressionModel = DataPredict.Models.QuantileRegression.new({QuantilesList = QuantilesList})

```

### Ordinal Regression Model Construction

```lua

-- ClassesList will use QuantilesList directly and you would be able to output raw numbers as classes for high probability predictions.

local OrdinalRegressionModel = DataPredict.Models.OrdinalRegression.new({ClassesList = QuantilesList})

```

## Upon Player Join

In here, what you need to do is:

* Store player data as a vector of numbers when the player purchases an item.

* Store the prices when the player purchases an item.

Below, we will show you how to create this:

```lua

-- We're just adding 1 here to add "bias".

local playerDataVector = {
    {
        1,
        numberOfCurrencyAmount,
        numberOfCurrencySpentInCurrentSession,
        numberOfCurrencySpentInAllSessions,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        numberOfItemsAmount,
        healthAmount
    }
}

-- Optionally, you can also concatenate this with the items' data.

local itemDataVector = {

  {rarity, timeSinceLastReleased}

}

-- This is our labelVector for QuantileRegressionModel.

local priceVector = {

    {price}

}

-- This is our labelVector for OrdinalRegressionModel.

local quantileVector = {

    {quantile}

}

```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to very high likelihood to purchase at a given price. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local numberOfQuantiles = #QuantilesList

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 8}, -100, 100) -- 100 random data with 8 features (including one "bias").

local priceVector = TensorL:createTensor({numberOfData, 1}, 9999) -- Making sure that at all values, it predicts very high price acceptance thresholds. Do not use math.huge here.

local quantileVector = TensorL:createTensor({numberOfData, numberOfQuantiles}, (1 / numberOfQuantiles)) -- Making sure that at all values, all predicted quantiles hold equal weights.

```

However, this require setting the models' parameters to these settings temporarily so that it can be biased at start up as shown below.

```lua

QuantileRegressionModel.maximumNumberOfIterations = 100

QuantileRegressionModel.learningRate = 0.3

OrdinalRegressionModel.maximumNumberOfIterations = 100

OrdinalRegressionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model.

```lua

local quantileRegressionCostArray = QuantileRegressionModel:train(playerDataVector, priceVector)

local ordinalRegressionCostArray = OrdinalRegressionModel:train(playerDataVector, quantileVector)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local QuantileRegressionModelParameters = QuantileRegressionModel:getModelParameters()

local OrdinalRegressionModelParameters = OrdinalRegressionModel:getModelParameters()

```

## Model Parameters Loading 

In here, we will use our model parameters so that it can be used to load out models. There are three cases in here:

1. The player is a first-time player.

2. The player is a returning player.

3. Every player uses the same global model.

### Case 1: The Player Is A First-Time Player

Under this case, this is a new player that plays the game for the first time. In this case, we do not know how this player would act.

We have a multiple way to handle this issue:

* We create a "global" model that trains from every player, and then make a deep copy of the model parameters and load it into our models.

* We take from other players' existing model parameters and load it into our models.

### Case 2: The Player Is A Returning Player

Under this case, you can continue using the existing model parameters that was saved in Roblox's Datastores.

```lua

QuantileRegressionModel:setModelParameters(QuantileRegressionModelParameters)

OrdinalRegressionModel:setModelParameters(OrdinalRegressionModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Prediction Handling

In order to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfCurrencySpentInCurrentSession, numberOfCurrencySpentInAllSessions, timePlayedInCurrentSession, timePlayedInAllSessions, numberOfItemsAmount, healthAmount}}

-- This is for Quantile Regression model.

local predictedQuantilePriceVector = QuantileRegressionModel:predict(currentPlayerDataVector)

-- If you're going for Bayesian Quantile Linear Regression model, please include the "quantilePriceVector" to the second parameter of predict() function.

local quantilePriceVector = {{0.25, 0.5, 0.75, 0.9}}

-- These values are are equivalent to the ones we set for Quantile Regression quantileList.

-- quantilePriceVector[1][1] = 25th percentile (conservative) price.
-- quantilePriceVector[1][2] = 50th percentile (balanced) price.
-- quantilePriceVector[1][3] = 75th percentile (aggressive) price.
-- quantilePriceVector[1][4] = 90th percentile (whale-focused) price.

local meanPriceVector, predictedQuantilePriceVector = QuantileRegressionModel:predict(currentPlayerDataVector, quantilePriceVector)

-- Then, we need the prediction from the ordinal regression to choose appropriate price.

local predictedQuantileVector = OrdinalRegressionModel:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can select the desired price by doing this:

```lua

local predictedQuantile = predictedQuantileVector[1][1] -- Getting our prediction from the OrdinalRegressionModel.

local priceIndex = table.find(QuantilesList, predictedQuantile)

local selectedPrice = predictedQuantilePriceVector[1][priceIndex] -- Getting our price from the QuantileRegressionModel.

```

That's all for today and see you later!
