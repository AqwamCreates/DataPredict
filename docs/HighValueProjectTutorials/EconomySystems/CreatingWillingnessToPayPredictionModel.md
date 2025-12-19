# [Economy Systems](../EconomySystems.md) - Creating Willingness-To-Pay Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict the likelihood of a player would purchase an item for a particular price.

Currently, you need these to produce the model:

* Bayesian linear regression model

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a quantile model. We have two algorithms that you can pick from.

| Model                               | Advantages                                                     | Disadvantages                                      |
|-------------------------------------|----------------------------------------------------------------|----------------------------------------------------|
| Bayesian Quantile Linear Regression | Data efficient and is computationally fast for small datasets. | Assume linear relationship and sensitive to noise. |
| Quantile Regression                 | Good against noisy data.                                       | Requires a lot of data.                            |

```lua

local DataPredict = require(DataPredict)

 -- This is required for Quantile Regression model, but not for Bayesian Quantile Linear Regression model.

local QuantilesList = {0.25, 0.5, 0.75, 0.90}

-- QuantilesList[1] = 25th percentile (conservative) price.
-- QuantilesList[2] = 50th percentile (balanced) price.
-- QuantilesList[3] = 75th percentile (aggressive) price.
-- QuantilesList[4] = 90th percentile (whale-focused) price.

local WillingnessToPayPredictionModel = DataPredict.Models.QuantileRegression.new({QuantilesList = QuantilesList})

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

* Store the prices when the player purchases an item.

Below, we will show you how to create this:

```lua

-- We're just adding 1 here to add "bias".

local playerDataVector = {
    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
}

-- Optionally, you can also concatenate this with the items' data.

local itemDataVector = {

  {rarity, timeSinceLastReleased}

}

-- This is our labelVector.

local priceVector = {

    {price}

}

```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to very high time-to-leave value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 6}, -100, 100) -- 100 random data with 6 features (including one "bias").

local labelDataMatrix = TensorL:createTensor({numberOfData, 1}, 9999) -- Making sure that at all values, it predicts very high time-to-leave value. Do not use math.huge here.

```

However, this require setting the model's parameters to these settings temporarily so that it can be biased to very high willingness-to-pay value at start up as shown below.

```lua

WillingnessToPayPredictionModel.maximumNumberOfIterations = 100

WillingnessToPayPredictionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model.

```lua

local costArray = WillingnessToPayPredictionModel:train(playerDataVector, priceVector)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = WillingnessToPayPredictionModel :getModelParameters()

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

WillingnessToPayPredictionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Prediction Handling

In order to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

-- This is for Quantile Regression model.

local predictedQuantilePriceVector = WillingnessToPayPredictionModel:predict(currentPlayerDataVector)

-- If you're going for Bayesian Quantile Linear Regression model, please include the "quantilePriceVector" to the second parameter of predict() function.

local quantilePriceVector = {{0.25, 0.5, 0.75, 0.9}}

-- These values are are equivalent to the ones we set for Quantile Regression quantileList.

-- quantilePriceVector[1][1] = 25th percentile (conservative) price.
-- quantilePriceVector[1][2] = 50th percentile (balanced) price.
-- quantilePriceVector[1][3] = 75th percentile (aggressive) price.
-- quantilePriceVector[1][4] = 90th percentile (whale-focused) price.

local meanPriceVector, predictedQuantilePriceVector = WillingnessToPayPredictionModel:predict(currentPlayerDataVector, quantilePriceVector)

```

Once you receive the predicted label vector, you can grab the pure number output and select desired price by doing this:

```lua

local conservativePrice = predictedQuantilePriceVector[1][1] -- 25th percentile
local balancedPrice = predictedQuantilePriceVector[1][2] -- 50th percentile (median)
local aggressivePrice = predictedQuantilePriceVector[1][3] -- 75th percentile  
local whalePrice = predictedQuantilePriceVector[1][4] -- 90th percentile

local playerEngagementLevel = timePlayedInAllSessions / 3600  -- hours played

local chosenPrice

if (playerEngagementLevel < 10) then

    chosenPrice = conservativePrice -- New players get cheaper prices.
    
elseif (playerEngagementLevel < 100) then

    chosenPrice = balancedPrice -- Regular players get median.
    
else

    chosenPrice = aggressivePrice -- Veteran players can afford more.
    
end
        
```

That's all for today and see you later!
