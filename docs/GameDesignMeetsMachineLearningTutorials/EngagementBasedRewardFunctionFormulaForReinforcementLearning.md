# Engagement-Based Reward Function Formula For Reinforcement Learning

## 


## Source Code

```lua

local optimalZoneMinimum = 0.5  -- Minimum reasonable thinking time.

local optimalZoneMaximum = 3.0  -- Maximum before likely distraction.

local cumulativeIncorrectActionPredictionCount = 0

local correctActionPrediction = 0

local totalActionPrediction = 0

local function calculateEngagementFromTiming(timeSinceLastAction)

	-- Shorter times = more engaged (bounded to prevent spam)

	if (timeSinceLastAction < optimalZoneMinimum) then

		return 0.8  -- Very engaged but possibly spammy

	elseif (timeSinceLastAction <= optimalZoneMaximum) then

		return 1.0  -- Perfect engagement (thinking but not distracted)

	else

		-- Exponential decay for long delays

		return math.exp(-(timeSinceLastAction - optimalZoneMaximum) / 5)

	end

end

local function learnFromPlayer(playerAction, durationBetweenAction)

	local isCorrectPredictedAction = (previousPredictedAction == playerAction)
	
	local engagementScore = calculateEngagementFromTiming(durationBetweenAction)
	
	local isPlayerLosing = getIsPlayerLosing()
	
	local baseReward = getBaseReward() -- Generic function for grabbing base reward relating to the game mechanic

	local correctActionPredictionValue = (isCorrectPredictedAction and 1) or 0
	
	if (isCorrectPredictedAction) then
		
		cumulativeIncorrectActionPredictionCount = 0
		
	else
		
		cumulativeIncorrectActionPredictionCount += 1
		
	end
	
	correctActionPrediction += correctActionPredictionValue 

	totalActionPrediction += 1
	
	local predictionAccuracyPercentage = correctActionPrediction / totalActionPrediction

	local reward = baseReward * predictionAccuracyPercentageComplement -- If it is getting too accurate, make the reward signal weaker.
	
	local varietyPenalty = 0
	
	if (cumulativeIncorrectActionPredictionCount > 1) then
		
		varietyPenalty = math.log(cumulativeIncorrectActionPredictionCount) -- The more it gets wrong, the more it gets punished.
		
	end

	reward *= engagementScore
	
	if (not isPlayerLosing) then reward -= varietyPenalty end -- Since we want the player to win, we must ignore the lack of variety in predictions.

	local predictedAction = EventModel:reinforce(stateVector, reward)

	return predictedAction

end

```
