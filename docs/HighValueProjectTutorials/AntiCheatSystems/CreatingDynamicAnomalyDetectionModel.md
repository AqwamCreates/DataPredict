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

## Designing Our State Vector

```lua

local stateVector = {{healthChangeAmount, damageAmount, hitStreakAmount}}

```

## Prediction Handling

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
