# Creating Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict the likelihood that the players will leave.

Currently, you need these to produce the model:

* One Class Support Vector Machine (Not Support Vector Machine)

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a classification model as shown below.

```lua

local DataPredict = require(DataPredict)

-- For single data point purposes, set the maximumNumberOfIterations to 1 to avoid overfitting. Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

-- Additionally, you must use RadialBasisFunction as the kernel function. This kernel accepts inputs of -infinity to infinity values, but outputs 0 to 1 values.

local AnomalyPredictionModel = DataPredict.Models.OneClassSupportVectorMachine.new({maximumNumberOfIterations = 1, learningRate = 0.3, kernelFunction = "RadialBasisFunction"})

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

Below, we will show you how to create this:

```lua

-- We're just adding 1 here to add "bias".

local playerDataVector = {
    {
        healthAmount,
        damageAmount
    }
}

```

If you want to add more data instead of relying on the initial data point, you actually can and this will improve the prediction accuracy. But keep in mind that this means you have to store more data. I recommend that for every 30 seconds, you store a new entry. Below, I will show how it is done.

```lua

local playerDataMatrix = {}
  
local snapshotIndex = 1
  
local function snapshotData()
  
 playerDataMatrix[snapshotIndex] = {

    healthAmount,
    damageAmount,

  }
  
  snapshotIndex = snapshotIndex + 1

end

```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to the "0" probability value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 2}, -100, 100) -- 100 random data with 2 features.

```

However, this require setting the model's parameters to these settings temporarily so that it can be biased to "0" at start up as shown below.

```lua

AnomalyPredictionModel.maximumNumberOfIterations = 100

AnomalyPredictionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model.

```lua

local costArray = AnomalyPredictionModel:train(playerDataVector)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = AnomalyPredictionModel:getModelParameters()

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

AnomalyPredictionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Prediction Handling

In other to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{healthAmount, damageAmount}}

local predictedLabelVector = AnomalyPredictionModel:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local anomalyProbability = (1 - predictedLabelVector[1][1])

```

So for the current session, you can determine what to do for the next session.

```lua

if (anomalyProbability >= 0.97) then -- Can be changed instead of 0.97.

--- Do a logic here to extend the play time for the next session. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create anomaly detection model that allows you to mark unusual activities. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
