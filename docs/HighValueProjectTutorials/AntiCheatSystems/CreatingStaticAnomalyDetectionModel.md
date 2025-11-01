# Creating Static Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create an anomaly-detection-based model that could detect unusual player behavior that often cheating or suspicious actions.

## Setting Up

Before we train our model, we will first need to construct a model, in which we have three approaches:

| Approach | Model                            | Properties                     | Notes                                                                                                                                         |
| -------- | -------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | Expectation-Maximization         | Incremental, Probabilistic     | Good for anomaly detection; can be updated with partial data, can produce multiple "normal" clusters (e.g. different player playstyles).          |
| 2        | One-Class Support Vector Machine | Non-Incremental, Kernel-Based  | Good for anomaly detection; cannot be updated real-time, heavier to run, best with best with "radial basis function" kernel.               |
| 3        | Gaussian Naive Bayes             | Incremental, Generative        | Not commonly used in anomaly detection; fast, can be updated with partial data, requires features to be independent (rare in real player data). |


### Approach 1: Expectation-Maximization

```lua

local DataPredict = require(DataPredict)

-- Number of clusters are highly dependent on how many playstyles you have. However, we recommend with 1 cluster as a starting point.

local AnomalyPredictionModel = DataPredict.Models.ExpectationMaximization.new({numberOfClusters = 1})

```

### Approach 2: One Class Support Vector Machine

```lua

local DataPredict = require(DataPredict)

-- You must use RadialBasisFunction as the kernel function. This kernel accepts inputs of -infinity to infinity values, but outputs 0 to 1 values.

local AnomalyPredictionModel = DataPredict.Models.OneClassSupportVectorMachine.new({maximumNumberOfIterations = 1, kernelFunction = "RadialBasisFunction"})

```

### Approach 3: Gaussian Naive Bayes

```lua

local DataPredict = require(DataPredict)

-- There are no parameters to set here.

local AnomalyPredictionModel = DataPredict.Models.GaussianNaiveBayes.new()

```

## Upon Player Join

In here, what you need to do is to store player data as a vector of numbers throughout the player's game session.

```lua

local playerDataMatrix = {}
  
local snapshotIndex = 1
  
local function snapshotData()
  
 playerDataMatrix[snapshotIndex] = {

    healthChangeAmount,
    damageAmount,
    hitStreakAmount,

  }
  
  snapshotIndex = snapshotIndex + 1

end

```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to the "1" probability value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 3}, -100, 100) -- 100 random data with 3 features.

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

local currentPlayerDataVector = {{healthChangeAmount, damageAmount, hitStreakAmount}}

local predictedLabelVector = AnomalyPredictionModel:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local isNormalProbability =  predictedLabelVector[1][1]

```

So for the current session, you can determine what to do for the next session.

```lua

if (isNormalProbability <= 0.03) then -- Can be changed instead of 0.03.

--- Do a logic here to deal with the player with the anomaly.

end

```

## Conclusion

This tutorial showed you on how to create anomaly detection model that allows you to detect unusual activities. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
