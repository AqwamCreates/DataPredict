# Creating Churn Red Zone Detection Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could detect if player is in the "red zone" before it leaves.

Currently, you need these to produce the model:

* Binary Regression

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below. First we need to actually understand how each of our model's binary function actually work.

| BinaryFunction   | Properties                                                        |
|------------------|-------------------------------------------------------------------|
| LogLo            | Best when red zone is very rare (< 20%); predicts early warnings. |
| Logistic         | Best for balanced or moderately rare events (20-45%).             |
| ComplementLogLog | Best when red zone is common (> 45%) - conservative detection     |

```lua

local DataPredict = require(DataPredict)

if (redZoneRatio <= 0.2) then

    binaryFunction = "LogLog"  -- Very rare events.
    
elseif redZoneRatio <= 0.45 then

    binaryFunction = "Logistic"  -- Moderately rare to balanced.
    
else

    binaryFunction = "ComplementLogLog"  -- Common events.
    
end

-- For single data point purposes, set the maximumNumberOfIterations to 1 to avoid overfitting. Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

local ChurnRedZoneDetectionModel = DataPredict.Models.BinaryRegression.new({maximumNumberOfIterations = 1, learningRate = 0.3, binaryFunction = binaryFunction})

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

Note that you must store a full set of data points for this to work. I recommend that for every 30 seconds, you store a new entry. Below, I will show how it is done.

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

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to very high time-to-leave value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 6}, -100, 100) -- 100 random data with 6 features (including one "bias").

local labelDataMatrix = TensorL:createTensor({numberOfData, 1}, 9999) -- Making sure that at all values, it predicts very high time-to-leave value. Do not use math.huge here.

```

However, this require setting the model's parameters to these settings temporarily so that it can be biased to very high time-to-leave value at start up as shown below.

```lua

ChurnRedZoneDetectionModel.maximumNumberOfIterations = 100

ChurnRedZoneDetectionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model. But first, we need to calculate the difference.

```lua

local playerLeftAtTime = os.time()

local numberOfRecordedTime = #recordedTimeArray

local numberOfRecordedTimeToMarkAsRedZone = numberOfRecordedTime * redZoneRatio

local startingIndexForRedZone = numberOfRecordedTime - numberOfRecordedTimeToMarkAsRedZone

local isInRedZoneVector = {}

local isInRedZone

for index, recordedTime in ipair(recordedTimeArray) do

    isInRedZone = (index >= startingIndexForRedZone)

    isInRedZoneVector[index] = {(isInRedZone and 1) or 0}

end

local costArray = ChurnRedZoneDetectionModel:train(playerDataMatrix, isInRedZoneVector)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = ChurnRedZoneDetectionModel:getModelParameters()

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

ChurnRedZoneDetectionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Prediction Handling

In other to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{1, numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

local predictedLabelVector = ChurnRedZoneDetectionModel:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local isPlayerInRedZone = predictedLabelVector[1][1]

```

We can do this for every 10 seconds and use this to extend the players' playtime by doing something like this:

```lua

if (isPlayerInRedZone >= 1) then

--- Do a logic here to extend the play time. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create "time to leave" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
