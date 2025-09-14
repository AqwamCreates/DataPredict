# Creating Time-To-Leave Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict when the player will leave.

Currently, you need these to produce the model:

* Any regression model

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

-- For single data point purposes, set the maximumNumberOfIterations to 1 to avoid overfitting. Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

local Regression = DataPredict.Models.LinearRegression.new({maximumNumberOfIterations = 1, learningRate = 0.3})

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

* Store the initial time that the player joined.

Below, we will show you how to create this:

```lua

-- We're just adding 1 here to add "bias".

local initialPlayerDataVector = {
    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
}

local initialJoinTime = os.time()

```

If you want to add more data instead of relying on the initial data point, you actually can and this will improve the prediction accuracy. But keep in mind that this means you have to store more data. I recommend that for every 30 seconds, you store a new entry. Below, I will show how it is done.

```lua

local initialPlayerDataMatrix = {}
  
local recordedTimeArray = {}
  
local snapshotIndex = 1
  
local function snapshotData()
  
 initialPlayerDataMatrix[snapshotIndex] = {

    1,
    numberOfCurrencyAmount,
    numberOfItemsAmount,
    timePlayedInCurrentSession,
    timePlayedInAllSessions,
    healthAmount

  }
  
  recordedTimeArray[snapshotIndex] = os.time()
  
  snapshotIndex = snapshotIndex + 1

end

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model. But first, we need to calculate the difference.

```lua

local timeToLeave = os.time() - initialJoinTime

local wrappedTimeToLeave = {

    {timeToLeave}

} -- Need to wrap this as our models can only accept matrices.

local costArray = Regression:train(initialPlayerDataVector, wrappedTimeToLeave)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = Regression:getModelParameters()

```

## Model Parameters Loading 

In here, we will use our model parameters so that it can be used to predict "time to leave". There are two cases in here:

1. The player is a first-time player.

2. The player is a returning player.

### Case 1: The Player Is A First-Time Player

Under this case, this is a new player that plays the game for the first time. In this case, we do not know how this player would act.

We have a multiple way to handle this issue:

* We create a "global" model that trains from every player, and then make a deep copy of the model parameters and load it into our models.

* We take from other players' existing model parameters and load it into our models.

### Case 2: The Player Is A Returning Player

Under this case, you can continue using the existing model parameters that was saved in Roblox's Datastores.

```lua

Regression:setModelParameters(ModelParameters)

```

## Prediction Handling

In other to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

local predictedLabelVector = Regression:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local timeToLeavePrediction = predictedLabelVector[1][1]

```

We can do this for every 10 seconds and use this to extend the players' playtime by doing something like this:

```lua

if (timeToLeavePrediction <= 60) then -- Can be changed instead of less than 60 seconds.

--- Do a logic here to extend the play time. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create "time to leave" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
