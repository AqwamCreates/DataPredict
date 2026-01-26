# Creating Probability-To-Leave Prediction Model

Hello guys! Today, I will be showing you on how to create a retention-based model that could predict the likelihood that the players will leave.

## Setting Up

Before we train our model, we will first need to choose a model as shown below.

| Model                           | Advantages                                                                                      | Disadvantages                                                                       |
|---------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Binary Regression               | Fast and simple.                                                                                | Requires a lot of data.                                                             |
| Factorization Machine           | Captures interaction between features; good for sparse data.                                    | Uses more computational resources.                                                  |
| Factorized Pairwise Interaction | Captures interaction between features; good for sparse data; faster than factorization machine. | Ignores linear combination between features and weights.                            |
| Neural Network                  | Captures complex patterns.                                                                      | Requires a lot of data; uses more computational resources as more layers are added. |

  * If you’re modeling the probability that a player leaves over time, Complementary Log–Log is often a better choice as a binary function output because it naturally models time-to-event processes where the probability of leaving increases asymmetrically as time passes.

```lua

local DataPredict = require(DataPredict)

-- For single data point purposes, set the maximumNumberOfIterations to 1 to avoid overfitting. Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

local LeavePredictionModel = DataPredict.Models.BinaryRegression.new({maximumNumberOfIterations = 1, learningRate = 0.3, binaryFunction = "ComplementaryLogLog"})

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

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to the "0" probability value. Then use this randomized dataset to pretrain the model before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 6}, -100, 100) -- 100 random data with 6 features (including one "bias").

local labelDataMatrix = TensorL:createTensor({numberOfData, 1}, 0) -- Making sure that at all values, it predicts zero probability of leaving.

```

However, this require setting the model's parameters to these settings temporarily so that it can be biased to "0" at start up as shown below.

```lua

LeavePredictionModel.maximumNumberOfIterations = 100

LeavePredictionModel.learningRate = 0.3

```

## Upon Player Leave

By the time the player leaves, it is time for us to train the model. But first, we need to calculate the difference.

```lua

local timeToLeave = os.time() - recordedTime

```

Currently, there are two ways to scale the probability.

1. Pure scaling

2. Sigmoid scaling

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

local costArray = LeavePredictionModel:train(playerDataVector, wrappedProbabilityToLeave)

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

local predictedLabelVector = LeavePredictionModel:predict(currentPlayerDataVector)

```

Once you receive the predicted label vector, you can grab the pure number output by doing this:

```lua

local timeToLeavePrediction = predictedLabelVector[1][1]

```

We can do this for every 10 seconds and use this to extend the players' playtime by doing something like this:

```lua

if (probabilityToLeavePrediction >= 0.97) then  -- Can be changed instead of 0.97.

--- Do a logic here to extend the play time. For example, bonus currency multiplier duration or random event.

end

```

## Conclusion

This tutorial showed you on how to create "probability to leave" prediction model that allows you to extend your players' playtime. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
