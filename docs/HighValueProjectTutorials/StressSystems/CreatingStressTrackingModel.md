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
        actionsPerMinute,
        effectiveActionsPerMinute,
    }
}

```

## Full Setup

```lua

local function onPlayerConnect(Player: Player)
	
	local warmUpTime = maximumWarmUpTime
	
	local hasMovedForTheFirstTime = false
	
	local isSuspiciousActivityDetected = false
	
	local suspicionScore = 0

	local rollingCost = 0

	local adaptiveMean = 0

	local adaptiveVariance = 0

	local adaptiveRate = 0.01 -- How fast thresholds adapt (lower = more stable).

	local timeSinceLastWarned = 0
	
	local Character: Model
	
	local Humanoid: Humanoid
	
	local previousStateVector

	local currentStateVector
	
	local previousOrientationY
	
	local isIdle

	local costArray

	local cost
	
	local valueDifference
	
	local standardDeviationValue
	
	local lowerBoundRollingCostThreshold
	
	local upperBoundRollingCostThreshold
	
	local deviationValue
	
	Player.CharacterAdded:Connect(function(NewCharacter)
		
		warmUpTime = maximumWarmUpTime
		
		hasMovedForTheFirstTime = false
		
		suspicionScore = 0
		
		rollingCost = 0
		
		cost = 0
		
		isSuspiciousActivityDetected = false
		
		Character = NewCharacter
		
		previousStateVector = getStateVector(Character)
		
		Humanoid = Character.Humanoid
		
		Humanoid.StateChanged:Connect(function(oldHumanoidEnum, newHumanoidEnum)
			
			if (newHumanoidEnum ~= humanoidRunningEnum) or (hasMovedForTheFirstTime) then return end
			
			hasMovedForTheFirstTime = true
			
		end)
		
	end)
	
	RunService.Heartbeat:Connect(function(delta)
		
		if (not Character) then return end
		
		currentStateVector = getStateVector(Character, previousStateVector, delta)
		
		isIdle = checkIfIsIdle(previousStateVector, currentStateVector)

		costArray = AnomalyDetectionModel:train(previousStateVector, currentStateVector)
		
		previousStateVector = currentStateVector
		
		if (warmUpTime > 0) and (hasMovedForTheFirstTime) then
			
			warmUpTime = warmUpTime - delta
			
			return
				
		elseif (not hasMovedForTheFirstTime) then
			
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
			
			suspicionScore = math.max(0, suspicionScore - 1)
		
		elseif (rollingCost < lowerBoundRollingCostThreshold) then
			
			deviationValue = (lowerBoundRollingCostThreshold - rollingCost)
			
		elseif (rollingCost > upperBoundRollingCostThreshold) then

			deviationValue = (rollingCost - upperBoundRollingCostThreshold)

		else

			suspicionScore = math.max(0, suspicionScore - 1)

		end
		
		suspicionScore = suspicionScore + (deviationValue * 0.1)
		
		suspicionScore = math.max(0, suspicionScore - (0.05 * (1 - math.abs(deviationValue))))
		
		isSuspiciousActivityDetected = (suspicionScore >= maximumSuspicionScore)
		
		SendDataRemoteEvent:FireClient(Player, isSuspiciousActivityDetected, suspicionScore, rollingCost, cost)
		
		if (isSuspiciousActivityDetected) and (timeSinceLastWarned <= 0) then
			
			timeSinceLastWarned = numberOfSecondsToResetCheatWarning
			
			warn(warningString:format(Player.Name, suspicionScore, rollingCost, cost))
			
		else
			
			timeSinceLastWarned = math.max(timeSinceLastWarned - delta, 0)
			
		end
	
	end)
	
end

```
