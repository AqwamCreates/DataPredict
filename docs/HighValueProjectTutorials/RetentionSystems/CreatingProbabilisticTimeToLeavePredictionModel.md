# [Retention Systems](../RetentionSystems.md) - Creating Probabilistic Time-To-Leave Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict when the player will leave.

Currently, you need these to produce the model:

* Bayesian linear regression model

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

local LeavePredictionModel = DataPredict.Models.BayesianLinearRegression.new()

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

* Store the initial time that the player joined.

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

local recordedTime = os.time()

```

If you want to add more data instead of relying on the initial data point, you actually can and this will improve the prediction accuracy. But keep in mind that this means you have to store more data. I recommend that for every 30 seconds, you store a new entry. Below, I will show how it is done.

```lua

local playerDataMatrix = {}
  
local recordedTimeArray = {}
  
local snapshotIndex = 1
  
local function snapshotData()
  
 playerDataMatrix[snapshotIndex] = {

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

local timeToLeave = os.time() - recordedTime

local wrappedTimeToLeave = {

    {timeToLeave}

} -- Need to wrap this as our models can only accept matrices.

local costArray = LeavePredictionModel:train(playerDataVector, wrappedTimeToLeave)

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

LeavePredictionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Prediction Handling

In other to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

-- Set the target time to leave to estimate their probabilities. Ensure that we have the current time as well.

local currentTime = os.time()

local expectedTimeToLeave1 = currentTime + 5

local expectedTimeToLeave2 = currentTime + 15

local expectedTimeToLeave3 = currentTime + 30

local expectedTimeToLeaveMatrix = {{expectedTimeToLeave1, expectedTimeToLeave2, expectedTimeToLeave3}}
        
local meanTimeToLeaveVector, probabilityToLeaveMatrix = LeavePredictionModel:predict(currentPlayerDataVector, expectedTimeToLeaveMatrix)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local meanTimeToLeave = meanTimeToLeaveVector[1][1]

local probability1 = probabilityToLeaveMatrix[1][1]

local probability2 = probabilityToLeaveMatrix[1][2]

local probability3 = probabilityToLeaveMatrix[1][3]
        
```

We can do this for every 10 seconds and use this to extend the players' playtime by doing something like this:

```lua

if (meanTimeToLeave <= 5) or (probability1 >= 0.5) or (probability2 >= 0.7) or (probability3 >= 0.9) then -- Can be changed instead of these values.

--- Do a logic here to extend the play time. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create "time to leave" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
