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

##

| Scenario Name          | Player State | AI Accuracy | AI Wrong Streak | Engagement | Final Reward Signal | AI Behavior & Lesson |
|------------------------|--------------|-------------|-----------------|------------|---------------------| :--- |
| The Rookie Phase       | Healthy      | Low (20%)   | 0               | High (1.0) | High Positive       | "Learn Fast!" <br> The AI is dumb, so the reward signal is strong. It rapidly learns the basic rules of the game. |
| The Glass Ceiling      | Healthy      | High (60%)  | 0               | High (1.0) | Low Positive        | "Stop Improving." <br> The AI is too smart. The `Complement` caps the reward, preventing it from learning to be perfect (unbeatable). |
| The Stubborn Mistake   | Healthy      | Medium      | 3-4             | High (1.0) | Negative            | "Change Strategy!" <br> Being wrong repeatedly hurts more than being right helps. The AI abandons its current guess immediately. |
| The Panic Spiral       | Healthy      | Low         | 10+             | Low (Spam) | Very Negative       | "Calm Down & Adapt." <br> Double punishment: The player is raging (low engagement multiplier) AND the AI is stubborn. Drastic behavior change required. |
| The Boredom Trap       | Healthy      | Medium      | 1               | Low (Idle) | Near Zero           | "Wake Them Up." <br> Even correct predictions yield almost no reward because the player is bored. The AI learns to trigger events to speed up gameplay. |
| The Critical Save      | Losing       | Any         | 0               | Any        | Negative            | "Save Them First." <br> Accuracy doesn't matter. If the player dies, the AI fails. It prioritizes survival mechanics over prediction. |
| The Fatal Stubbornness | Losing       | Any         | 3+              | Any        | Huge Negative       | "Never Do This Again." <br> The worst possible outcome. The AI was wrong repeatedly WHILE the player died. Maximum punishment ensures this pattern is erased. |
| The Lucky Recovery     | Healthy      | Low         | 0 (Reset)       | Medium     | Medium Positive     | "Good Adaptation." <br> After a streak of errors, the AI finally guessed right. The streak reset removes the penalty, reinforcing the new strategy. |
| The Noise Filter       | Healthy      | Medium      | 1               | High       | Positive            | "Ignore Glitches." <br> One wrong guess is treated as noise. No penalty applied, preventing the AI from overreacting to random player quirks. |
| The Flow State         | Healthy      | Medium      | 0               | High (1.0) | High Positive       | "Maintain Status Quo." <br> The player is happy, the AI is fair. The system stabilizes here, making minimal changes to keep the player in the zone. |

