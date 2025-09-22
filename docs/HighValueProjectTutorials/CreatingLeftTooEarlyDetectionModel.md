# Creating Left-To-Early Detection Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict the likelihood that the players will leave.

Currently, you need these to produce the model:

* Support Vector Machine

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a classification model as shown below.

```lua

local DataPredict = require(DataPredict)

-- For single data point purposes, set the maximumNumberOfIterations to 1 to avoid overfitting. Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

-- Additionally, you must use RadialBasisFunction as the kernel function. This kernel accepts inputs of -infinity to infinity values, but outputs 0 to 1 values.

local LeftToEarlyPredictionModel = DataPredict.Models.SupportVectorMachine.new({maximumNumberOfIterations = 1, kernelFunction = "RadialBasisFunction"})

```

## Upon Player Join

In here, what you need to do is:

* Store initial player data as a vector of numbers.

* Store the initial time that the player joined.

Below, we will show you how to create this:

```lua


local playerDataVector = {
    {
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

local playerDataMatrix = {}
  
local recordedTimeArray = {}
  
local snapshotIndex = 1
  
local function snapshotData()
  
 playerDataMatrix[snapshotIndex] = {

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

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to the "0" probability value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 5}, -100, 100) -- 100 random data with 5 features.

local labelDataMatrix = TensorL:createTensor({numberOfData, 1}, 0) -- Making sure that at all values, it predicts zero probability of leaving.

```

However, this require setting the model's parameters to these settings temporarily so that it can be biased to "0" at start up as shown below.

```lua

LeftToEarlyPredictionModel.maximumNumberOfIterations = 100

LeftToEarlyPredictionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model. But first, we need to calculate the difference.

```lua

local timeElapsed = os.time() - initialJoinTime

```

Currently, there are two ways to scale the probability.

1. Pure scaling

2. Sigmoid scaling

### Method 1: Pure Scaling

### Method 1: Pure Scaling

```lua

local probabilityToLeave = 1 / timeToLeave

```

### Method 2: Sigmoid Scaling

```lua

-- Large scaleFactor means slower growth. scaleFactor should be based on empirical average session length.

local probabilityToLeave = math.exp(-timeToLeave / scaleFactor)

```

Once you have chosen to scale your values, we must do this:

```lua

local wrappedProbabilityToLeave = {

    {probabilityToLeave}

} -- Need to wrap this as our models can only accept matrices.

local costArray = LeftToEarlyPredictionModel:train(playerDataVector, wrappedProbabilityToLeave)

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

local predictedProbabilityToLeave = predictedLabelVector[1][1]

```

So for the current session, you can determine what to do for the next session. Notice when "predictedProbabilityToLeave" is low, but the player has already left. This contradiction means that the model didn't expect the player left too soon.

```lua

if (predictedProbabilityToLeave <= 0.1) then -- Can be changed instead of 0.97.

--- Do a logic here to extend the play time for the next session. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create "left too early" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
