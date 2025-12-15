# [Retention Systems](../RetentionSystems.md) - Creating Willingness-To-Pay Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict the likelihood of a player would purchase an item for a particular price.

Currently, you need these to produce the model:

* Bayesian linear regression model

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

local WillingnessToPayPredictionModel = DataPredict.Models.BayesianLinearRegression.new()

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

However, this require setting the model's parameters to these settings temporarily so that it can be biased to very high time-to-leave value at start up as shown below.

```lua

WillingnessToPayPredictionModel.maximumNumberOfIterations = 100

WillingnessToPayPredictionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model.

```lua

local costArray = WillingnessToPayPredictionModel:train(playerDataVector, wrappedTimeToLeave)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = LeavePredictionModel:getModelParameters()

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

In other to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

local quantilePriceVector = {{0.25, 0.5, 0.75, 0.9}}

-- quantilePrices[1][1] = 25th percentile (conservative) price
-- quantilePrices[1][2] = Median price
-- quantilePrices[1][3] = 75th percentile (aggressive) price  
-- quantilePrices[1][4] = 90th percentile (whale-focused) price

local meanPrice, quantilePrices = model:predict(currentPlayerDataVector, quantilePriceVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local meanTimeToLeave = meanTimeToLeaveVector[1][1]
        
```

We can do this for every 10 seconds and use this to extend the players' playtime by doing something like this:

```lua


```

## Conclusion

This tutorial showed you on how to create "time to leave" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
