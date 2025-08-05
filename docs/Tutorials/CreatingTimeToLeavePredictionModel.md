# Creating Time To Leave Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict when the player will leave.

Currently, you need these to produce the model:

* Any regression models

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

local Regression = DataPredict.Models.LinearRegression.new({learningRate = 0.3}) -- Ensure that the learningRate is not too high or too low.

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

* Store the initial time that the player joined.

Below, we will show you how to create this:

```lua

-- We're just adding 1 here to add "bias".

local initialPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

local initialJoinTime = os.time()

```

Additionally, if a model has been trained as this player is a returning player, load the model parameters from Roblox's Datastores

```lua

Regression:getModelParameters(ModelParameters)

```

If you want to add more data instead of relying on the initial data point, you can! But keep in mind that this means you have to store more data. I recommend that for every 30 seconds, you store a new entry. Below, I will show how it is done.

```lua

local initialPlayerDataVector = {}

local recordedTimeArray = {}

initialPlayerDataVector[1] = {1, numberOfCurrencyAmount1, numberOfItemsAmount1, timePlayedInCurrentSession1, timePlayedInAllSessions1, healthAmount1}

recordedTimeArray[1] = os.time()

initialPlayerDataVector[2] = {1, numberOfCurrencyAmount2, numberOfItemsAmount2, timePlayedInCurrentSession2, timePlayedInAllSessions2, healthAmount2}

recordedTimeArray[2] = os.time()

initialPlayerDataVector[3] = {1, numberOfCurrencyAmount3, numberOfItemsAmount3, timePlayedInCurrentSession3, timePlayedInAllSessions3, healthAmount3}

recordedTimeArray[3] = os.time()

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model. But first, we need to calculate the difference.

```lua

local timeToLeave = initialJoinTime - os.time()

local wrappedTimeToLeave = {{timeToLeave}} -- Need to wrap this as our models can only accept matrices.

local costArray = Regression:train(initialPlayerDataVector, wrappedTimeToLeave)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.



## Upon Player In-Game

There are two cases in here:

1. Player is a first-time player.

2. Player is a returning player.

### Case 1: Player A First-Time Player

Under this case, this is a new player that plays the game for the first time. In this case, we do not know how this player would act.

We have a multiple way to handle this issue.

* We create a "global" model that trains from every players, and then make a deep copy of the model parameters and load it into our models.

* We take from other player's existing model parameters and load it into our models.

### Case 1: Player A First-Time Player
