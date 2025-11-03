# Creating Dynamic Anomaly Detection Model

Hello guys! Today, I will be showing you on how to create an anomaly-detection-based model that could detect unusual player behavior that often cheating or suspicious actions.

Additionally, a demo file for more advanced version of this anti cheat model can be found [here](https://github.com/AqwamCreates/DataPredict-Tutorials-Source-Codes/blob/main/Dynamic%20Anti%20Cheat/Dynamic%20Anti%20Cheat.rbxl).

## Setting Up

Before we train our model, we will first need to construct a model, in which we have three approaches:

| Approach | Model                                          | Advantages                              | Disadvantages                            |
| -------- | ---------------------------------------------- | --------------------------------------- | ---------------------------------------- |
| 1        | Kalman Filter                                  | Good against noisy data.                | Assumes values are linear.               |
| 2        | Unscented Kalman Filter                        | Good against noisy data.                | Requires some parameter configurations.  |
| 3        | Unscented Kalman Filter (DataPredict Variant)  | Same as 2, but more numerically stable. | Same as 2, but slower.                   |
| 4        | Dynamic Bayesian Network                       | Extremely fast.                         | Assumes values are normally distributed. |

### Approach 1 - 3: Kalman Filter And Its Variants

```lua

local DataPredict = require(DataPredict)

--[[

  For best results, you must set "lossFunction" to "Mahalanobis" for anomaly detection.

  You can also set "useJosephForm" to false if you want a more faster calculation by trading numerical stability and accuracy.

  By default, "lossFunction" is set to "L2" and "useJosephForm" is set to true.

  The original "Unscented Kalman Filter" does not have "useJosephForm".

--]]

local AnomalyDetectionModel = DataPredict.Models.KalmanFilter.new({lossFunction = "Mahalanobis", useJosephForm = true})

```

### Approach 4: Dynamic Bayesian Network

```lua

local DataPredict = require(DataPredict)

-- There are no parameters to set here.

local AnomalyDetectionModel = DataPredict.Models.DynamicBayesianNetwork.new()

```

### Optional: What About the Extended Kalman Filter?

You might have heard of the Extended Kalman Filter, which uses Jacobian matrices to handle non-linearities. However, this version often requires more setup (derivative functions) and can be less stable than the Unscented version. 

For most Roblox or real-time anomaly detection cases, the Unscented Kalman Filter provides better results with less tuning effort.

## Designing Our State Vector

```lua

local stateVector = {{healthChangeAmount, damageAmount, killPerDurationFromLastKill}}

```

### Note: Roblox Server Use Cases

When using Kalman Filter and its variants, you can only use up to 7 features before it causes the Roblox server to freeze. This is because these models require high computational resources. 

## Anomaly Detection

Before we get through the code, we first need to understand on how to set a proper cost threshold to filter out unusual activities.

* You must make sure to use multiple costs to check suspicious behaviour. This is because a single step cost mostly comes from random noise (e.g. network latency, CPU / GPU clocking issues and so on).

* When you set the aggregate cost threshold very high, this will detect the "blatant" cheating.

* When you set the aggregate cost threshold to very low, this will detect the "expert" cheating. However, you need to be careful since the idle state can produce very low cost.

* Between these two aggregate cost threshold, the cost generated is as a result of players' noisy, but consistent movements.

Therefore, it is important for you to implement this model and test it under non-cheating circumstances to get these cost threshold.

```lua

-- Do not use these cost threshold values. These are extremely specific to your model's setup.

local lowerBoundRollingCostThreshold = 1

local upperBoundRollingCostThreshold = 3

local rollingCostRate = 0.9

local rollingCostRateComplement = (1 - rollingCostRate)

local maximumSuspicionCount = 30

local function run(Player)

    local suspicionCount = 0

    local rollingCost = 0

    local isPlayerInServer = true

    local previousStateVector = getStateVector(Player)

    local currentStateVector

    local isIdle

    local costArray

    local cost

    while isPlayerInServer do

        currentStateVector = getStateVector(Player)

        isIdle = checkIfIsIdle(previousStateVector, currentStateVector)

        costArray = AnomalyDetectionModel:train(previousStateVector, currentStateVector)

        cost = costArray[1]

        previousStateVector = currentStateVector

        -- Check if the player is performing suspiciously.

         rollingCost = (rollingCostRate * rollingCost) + (rollingCostRateComplement * cost) -- Exponential smoothing.

        if (isIdle) then
        
            suspicionCount = math.max(0, suspicionCount - 1)

        elseif (rollingCost < lowerBoundRollingCostThreshold) or (rollingCost > upperBoundRollingCostThreshold) then
        
            suspicionCount = suspicionCount + 1
            
        else
        
            suspicionCount = math.max(0, suspicionCount - 1)
            
        end
        
        if (suspicionCount >= maximumSuspicionCount) then kickPlayer(Player) end

        task.wait()

        isPlayerInServer = checkIfPlayerIsInServer(Player)

    end

end

```

## Conclusion

This tutorial showed you on how to create anomaly detection model that allows you to detect unusual activities. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
