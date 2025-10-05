# Creating Labelless Left-To-Early Detection Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict the likelihood that the players will leave.

## Setting Up

Before we train our model, we will first need to construct a model. Currently we have two approaches:

* Approach 1: Gausian Naive Bayes (Can Be Incremental. Requires Features To Be Independent To Each Other.)

  * Feature independence is not guaranteed with players' data.

* Approach 2: One-Class Support Vector Machine (Non-Incremental.)

## Approach 1

```lua

local DataPredict = require(DataPredict)

-- There's no parameters to set up here.

local LeftToEarlyPredictionModel = DataPredict.Models.GaussianNaiveBayes.new()

```

## Approach 2

```lua

local DataPredict = require(DataPredict)

-- You must use RadialBasisFunction as the kernel function. This kernel accepts inputs of -infinity to infinity values, but outputs 0 to 1 values.

-- Additionally, set your beta value to 1. This tells the model that all the data belongs the positive class (the time that the player is currently in session).

local LeftToEarlyPredictionModel = DataPredict.Models.OneClassSupportVectorMachine.new({maximumNumberOfIterations = 1, kernelFunction = "RadialBasisFunction", beta = 1})

```

## Upon Player Join

In here, what you need to do is to store player data as a vector of numbers throughout the session.

Below, we will show you how to create this:

```lua

local playerDataMatrix = {}
  
local snapshotIndex = 1
  
local function snapshotData()
  
 playerDataMatrix[snapshotIndex] = {

    numberOfCurrencyAmount,
    numberOfItemsAmount,
    timePlayedInCurrentSession,
    timePlayedInAllSessions,
    healthAmount

  }
  
  snapshotIndex = snapshotIndex + 1

end

```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to the "0" probability value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 5}, -100, 100) -- 100 random data with 5 features.

```

However, this require setting the model's parameters to these settings temporarily so that it can be biased to "0" at start up as shown below.

```lua

LeftToEarlyPredictionModel.maximumNumberOfIterations = 100

LeftToEarlyPredictionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model.

```lua

local costArray = LeftToEarlyPredictionModel:train(playerDataVector)

```

This should give you a model that predicts a rough estimate when they'll leave.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = LeftToEarlyPredictionModel:getModelParameters()

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

LeftToEarlyPredictionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Prediction Handling

In other to produce predictions from our model, we must perform this operation:

```lua

local currentPlayerDataVector = {{numberOfCurrencyAmount, numberOfItemsAmount, timePlayedInCurrentSession, timePlayedInAllSessions, healthAmount}}

local predictedLabelVector = LeftToEarlyPredictionModel:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local predictedToStayProbability = predictedLabelVector[1][1]

```

So for the current session, you can determine what to do for the next session. Notice how high "stay proability" is high, but the player has already left. This contradiction means that the model didn't expect the player left too soon.

```lua

if (stayingProbability >= 0.97) then -- Can be changed instead of 0.97.

--- Do a logic here to extend the play time for the next session. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create "left too early" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
