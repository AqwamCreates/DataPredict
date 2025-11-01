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

  By default, it is set to true. 

--]]

local AnomalyPredictionModel = DataPredict.Models.KalmanFilter.new({useJosephForm = true})

```

### Approach 2: Dynamic Bayesian Network

```lua

local DataPredict = require(DataPredict)

-- There are no parameters to set here.

local AnomalyPredictionModel = DataPredict.Models.DynamicBayesianNetwork.new()

```

## Designing Our State Vector

```lua

local stateVector = {{healthChangeAmount, damageAmount, hitStreakAmount}}

```

## Anomaly Detection

In order for us to check for unusual activities, we will have to rely on the training cost by calling train() function.

But first, we need to understand on how to set a proper cost threshold to filter out unusual activities.

* When you set the cost threshold very high, this will detect the "blatant" cheating.

* When you set the cost threshold to very low, this will detect the "expert" cheating. However, you need to be careful since the idle state can produce very low cost.

* Between these two cost threshold, the cost generated is as a result of players' noisy, but consistent movements.

Therefore, it is important for you to implement this model and test it under non-cheating circumstances to get these cost threshold.

```lua



```

## Conclusion

This tutorial showed you on how to create anomaly detection model that allows you to mark unusual activities. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
