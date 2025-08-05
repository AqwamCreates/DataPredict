# Creating Time To Leave Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict when the player will leave.

Currently, you need these to produce the model:

* Any regression models

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

local Regression = DataPredict.Models.LinearRegression.new({learningRate = 0.1}) -- Ensure that the learningRate is not too high or too low.

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

* Store the initial time that the player joined.

Below, we will show you how to create this:

```lua

local initialPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

local initialJoinTime = os.time()

```

Additionally, if a model has been trained as this player is a returning player, load the model parameters from Roblox's Datastores

```

Regression:getModelParameters(ModelParameters)

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model. But first, we need to calculate the difference.

```

local timeToLeave = initialJoinTime - os.time()

local wrappedTimeToLeave = {{timeToLeave}} -- Need to wrap this as our models can only accept matrices.

local costArray = Regression:train(initialPlayerDataVector, wrappedTimeToLeave)

```

This should give you a model that predicts a rough estimate when they'll leave.
