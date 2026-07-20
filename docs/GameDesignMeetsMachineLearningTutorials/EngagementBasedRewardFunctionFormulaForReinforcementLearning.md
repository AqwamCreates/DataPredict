# Engagement-Based Reward Function Formula For Reinforcement Learning

## Priority List For Engagement

* Priority 1: Be merciful (Retention)

* Priority 2: Be fair (Limiter)

* Priority 3: Be adaptable (Variety)

* Priority 4: Be accurate (Base Reward)

## Component Requirements For Reward Function

| Reward Function Component                            | Reason                                                                            | How It Produces Human-Like Behaviours                                                                                                                                                                                                      |
|------------------------------------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Current Engagement Score                             | To evaluate the players' flow state.                                              | Acts as a baseline for the reward function. Helps the AI to make accurate decisions based on the players' engagement.                                                                                                                      |
| Overall Accuracy Complement Score / Inaccuracy Score | The more accurate the AI overall, the more frustrating the AI is for the players. | As a beginner, the AI have a lot of things to learn from the player leads to high reward. Eventually, the AI will be at professional level relative to the player, where the AI will no longer learn much from the player.                 |
| Cumulative Incorrect Action Prediction Count         | To make AI adaptive.                                                              | The AI will first think that the first few wrongs as coincidence, but eventually realizes the current strategy will not work. After realizing the strategy does not work, the punishment plateaus as the AI is expected to lose from here. |
| Survival Override (Loss State)                       | To prioritize player retention over prediction accuracy.                          | If the player is about to lose, the reward flips negative immediately. This teaches the AI that saving the player is more important than being right. This is because players generally have high chance of leaving when the player loses. |

## Source Code

```lua

local optimalZoneMinimumInSeconds = 0.5  -- Minimum reasonable thinking time.

local optimalZoneMaximumInSeconds = 3.0  -- Maximum before likely distraction.

local meanDurationFromThinkingToActionInSeconds = 5 -- Dependent on your game.

local varietyPenaltyScale = 1 -- Dependent on your model.

local cumulativeIncorrectActionPredictionCount = 0

local correctActionPrediction = 0

local totalActionPrediction = 0

local function calculateEngagementFromTiming(timeSinceLastAction)

	-- Shorter times = more engaged (bounded to prevent spam).

	if (timeSinceLastAction < optimalZoneMinimumInSeconds) then

		return 0.8  -- Very engaged but possibly spamming.

	elseif (timeSinceLastAction <= optimalZoneMaximumInSeconds) then

		return 1.0  -- Perfect engagement (thinking but not distracted).

	else

		-- Exponential decay for long delays.

		return math.exp(-(timeSinceLastAction - optimalZoneMaximumInSeconds) / meanDurationFromThinkingToActionInSeconds)

	end

end

local function learnFromPlayer(playerAction, durationBetweenAction)

	local isCorrectPredictedAction = (previousPredictedAction == playerAction)
	
	local engagementScore = calculateEngagementFromTiming(durationBetweenAction)
	
	local isPlayerLosing = getIsPlayerLosing() -- This is a general function for grabbing the player's losing status.
	
	local baseReward = getBaseReward() -- This is a general function for grabbing base reward relating to the game's mechanics.

	local correctActionPredictionValue = (isCorrectPredictedAction and 1) or 0
	
	if (isCorrectPredictedAction) then
		
		cumulativeIncorrectActionPredictionCount = 0
		
	else
		
		cumulativeIncorrectActionPredictionCount += 1
		
	end
	
	correctActionPrediction += correctActionPredictionValue 

	totalActionPrediction += 1
	
	local predictionAccuracyPercentage = correctActionPrediction / totalActionPrediction

	local predictionAccuracyPercentageComplement = 1 - predictionAccuracyPercentage

	local reward = baseReward * predictionAccuracyPercentageComplement -- If it is getting too accurate, make the reward signal weaker.
	
	local varietyPenalty = 0
	
	if (cumulativeIncorrectActionPredictionCount > 1) then
		
		varietyPenalty = varietyPenaltyScale * math.log(cumulativeIncorrectActionPredictionCount) -- The more it gets wrong, the more it gets punished.
		
	end

	reward *= engagementScore

	if (isPlayerLosing) then 
		
		reward = -reward -- When the player is losing, the reward is negative because players generally have high chance of leaving. Hence, we need to prioritize the retention over the accuracy.
		
	else
		
		reward -= varietyPenalty -- Since we want the player to live, we must ignore the lack of variety in predictions.
		
	end

	local predictedAction = EventModel:reinforce(stateVector, reward)

	return predictedAction

end

```
