# [Stress Systems](../StressSystems.md) - Creating Stress Tracking Model

Hello guys! Today, I will be showing you on how to create a stress-tracking model to determine the players' stress level.

Currently, you need these to produce the model:

* Any KalmanFilter model

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

local StressTrac = DataPredict.Models.GammaRegression.new({maximumNumberOfIterations = 1, learningRate = 0.3})

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
        actionsPerMinute,
        effectiveActionsPerMinute,
        healthAmount
    }
}

local recordedTime = os.time()

```

