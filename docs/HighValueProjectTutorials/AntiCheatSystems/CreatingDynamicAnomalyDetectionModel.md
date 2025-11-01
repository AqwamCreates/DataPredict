# Creating Dynamic Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create a anomaly-detection-based model that could detect unusual player behavior that often cheating or suspicious actions.

## Setting Up

Before we train our model, we will first need to construct a model, in which we have three approaches:

| Approach | Model                    | Notes                    |
| -------- | -------------------------| ------------------------ |
| 1        | Kalman Filter            | Good against noisy data. |
| 2        | Dynamic Bayesian Network | Extremely fast.          |

### Approach 1: Kalman Filter

```lua

local DataPredict = require(DataPredict)

--[[

  You can set "useJosephForm" to false if you want a more faster calculation by trading numerical stability and accuracy.

  By default, "useJosephForm" is set to true. 

--]]

local AnomalyPredictionModel = DataPredict.Models.KalmanFilter.new({useJosephForm = true})

```

### Approach 2: Dynamic Bayesian Network

```lua

local DataPredict = require(DataPredict)

-- There are no parameters to set here.

local AnomalyPredictionModel = DataPredict.Models.DynamicBayesianNetwork.new()

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

This tutorial showed you on how to create anomaly detection model that allows you to mark unusual activities. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
