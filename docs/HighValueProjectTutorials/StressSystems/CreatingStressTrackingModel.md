# [Stress Systems](../StressSystems.md) - Creating Stress Tracking Model

Hello guys! Today, I will be showing you on how to create a stress-tracking model to determine the players' stress level.

Currently, you need these to produce the model:

* Any KalmanFilter model

* A player data that is stored in matrix

## Setting Up

Before we train our model, we will first need to construct a regression model as shown below.

```lua

local DataPredict = require(DataPredict)

local StressTrackingModel = DataPredict.Models.KalmanFilter.new({})

```

## The Feature Matrix

```lua

-- We're just adding 1 here to add "bias".

local playerDataVector = {
    {
        1,
		timeSinceLastInput,
        actionsPerMinute,
        effectiveActionsPerMinute,
    }
}

```

## Full Setup

```lua

local maximumStressScore

local function onPlayerConnect(Player: Player)
	
	local hasMovedForTheFirstTime = false
	
	local isStressDetected = false
	
	local stressScore = 0

	local rollingCost = 0

	local adaptiveMean = 0

	local adaptiveVariance = 0

	local adaptiveRate = 0.01 -- How fast thresholds adapt (lower = more stable).

	local timeSinceLastWarned = 0
	
	local previousStateVector

	local currentStateVector
	
	local isIdle

	local costArray

	local cost
	
	local valueDifference
	
	local standardDeviationValue
	
	local lowerBoundRollingCostThreshold
	
	local upperBoundRollingCostThreshold
	
	local deviationValue

	InputRemoteEvent.OnClientEvent:Connect(function(delta)
		
		if (not Character) then return end
		
		currentStateVector = getStateVector(Character, previousStateVector, delta)
		
		isIdle = checkIfIsIdle(previousStateVector, currentStateVector)

		costArray = AnomalyDetectionModel:train(previousStateVector, currentStateVector)
		
		previousStateVector = currentStateVector
		
		if (not hasMovedForTheFirstTime) then
			
			return
			
		end

		cost = costArray[1]
		
		rollingCost = (rollingCostRate * rollingCost) + (rollingCostRateComplement * cost) -- Exponential smoothing.
		
		valueDifference = rollingCost - adaptiveMean
		
		adaptiveMean = adaptiveMean + (adaptiveRate * valueDifference)
		
		adaptiveVariance = (1 - adaptiveRate) * (adaptiveVariance + adaptiveRate * math.pow(valueDifference, 2))

		standardDeviationValue = math.sqrt(adaptiveVariance)
		
		lowerBoundRollingCostThreshold = adaptiveMean - 3 * standardDeviationValue
		
		upperBoundRollingCostThreshold = adaptiveMean + 3 * standardDeviationValue
		
		deviationValue = 0
		
		if (isIdle) then
			
			stressScore = math.max(0, stressScore - 1)
		
		elseif (rollingCost < lowerBoundRollingCostThreshold) then
			
			deviationValue = (lowerBoundRollingCostThreshold - rollingCost)
			
		elseif (rollingCost > upperBoundRollingCostThreshold) then

			deviationValue = (rollingCost - upperBoundRollingCostThreshold)

		else

			stressScore = math.max(0, stressScore - 1)

		end
		
		stressScore = stressScore + (deviationValue * 0.1)
		
		stressScore = math.max(0, stressScore - (0.05 * (1 - math.abs(deviationValue))))
		
		isStressDetected = (stressScore >= maximumStressScore)
		
		SendDataRemoteEvent:FireClient(Player, isStressDetected, stressScore, rollingCost, cost)
		
		if (isStressDetected) and (timeSinceLastWarned <= 0) then
			
			timeSinceLastWarned = numberOfSecondsToResetCheatWarning
			
			warn(warningString:format(Player.Name, stressScore, rollingCost, cost))
			
		else
			
			timeSinceLastWarned = math.max(timeSinceLastWarned - delta, 0)
			
		end
	
	end)
	
end

```
