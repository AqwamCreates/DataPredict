# Creating Dynamic Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create an anomaly-detection-based model that could detect unusual player behavior that often cheating or suspicious actions.

## Setting Up

Before we train our model, we will first need to construct a model, in which we have three approaches:

| Approach | Model                                          | Advantages               | Disadvantages                            |
| -------- | ---------------------------------------------- | ------------------------ | ---------------------------------------- |
| 1        | Kalman Filter                                  | Good against noisy data. | Assumes values are linear.               |
| 2        | Unscented Kalman Filter (DataPredict Variant)  | Good against noisy data. | Requires some parameter configurations.  |
| 3        | Dynamic Bayesian Network                       | Extremely fast.          | Assumes values are normally distributed. |

### Approach 1 / 2: Kalman Filter

```lua

local DataPredict = require(DataPredict)

--[[

  For best results, set "lossFunction" to "Mahalanobis" for anomaly detection.

  You can also set "useJosephForm" to false if you want a more faster calculation by trading numerical stability and accuracy.

  By default, "lossFunction" is set to "L2" and "useJosephForm" is set to true.

--]]

local AnomalyPredictionModel = DataPredict.Models.KalmanFilter.new({lossFunction = "Mahalanobis", useJosephForm = true})

```

### Approach 3: Dynamic Bayesian Network

```lua

local DataPredict = require(DataPredict)

-- There are no parameters to set here.

local AnomalyPredictionModel = DataPredict.Models.DynamicBayesianNetwork.new()

```

### Optional: What About the Extended Kalman Filter?

You might have heard of the Extended Kalman Filter, which uses Jacobian matrices to handle non-linearities. However, this version often requires more setup (derivative functions) and can be less stable than the Unscented version. 

For most Roblox or real-time anomaly detection cases, the Unscented Kalman Filter provides better results with less tuning effort.

## Designing Our State Vector

```lua

local stateVector = {{healthChangeAmount, damageAmount, killPerDurationFromLastKill}}

```

## Anomaly Detection

In order for us to check for unusual activities, we will have to rely on the training cost by calling train() function.

But first, we need to understand on how to set a proper cost threshold to filter out unusual activities.

* When you set the cost threshold very high, this will detect the "blatant" cheating.

* When you set the cost threshold to very low, this will detect the "expert" cheating. However, you need to be careful since the idle state can produce very low cost.

* Between these two cost threshold, the cost generated is as a result of players' noisy, but consistent movements.

Therefore, it is important for you to implement this model and test it under non-cheating circumstances to get these cost threshold.

```lua

-- Do not use these cost threshold values. These are extremely specific to your model's setup.

local lowerBoundCostThreshold = 1

local upperBoundCostThreshold = 3

local function run(Player)

    local isPlayerInServer = true

    local previousStateVector = getStateVector(Player)

    local currentStateVector

    local costArray

    local cost

    while isPlayerInServer do

        currentStateVector = getPlayerDataVector(Player)
    
        costArray = AnomalyPredictionModel:train(previousStateVector, currentStateVector)

        cost = costArray[1]

        previousStateVector = currentStateVector

        -- Checks if the player is performing suspiciously.

        if (cost < lowerBoundCostThreshold) or (cost > upperBoundCostThreshold) then

          kickPlayer(Player)

        end

        task.wait()

        isPlayerInServer = checkIfPlayerIsInServer(Player)

    end

end

```

## Conclusion

This tutorial showed you on how to create anomaly detection model that allows you to detect unusual activities. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
