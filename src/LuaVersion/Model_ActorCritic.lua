--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningActorCriticNeuralNetworkBaseModel = require("Model_ReinforcementLearningActorCriticNeuralNetworkBaseModel")

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)

local function sampleAction(actionProbabilityVector)

	local totalProbability = 0

	for _, probability in ipairs(actionProbabilityVector[1]) do

		totalProbability += probability

	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0

	local actionIndex = 1

	for i, probability in ipairs(actionProbabilityVector[1]) do

		cumulativeProbability += probability

		if (randomValue > cumulativeProbability) then continue end

		actionIndex = i

		break

	end

	return actionIndex

end

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

function ActorCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	local NewActorCriticModel = ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	local actionProbabilityHistory = {}
	
	local criticValueHistory = {}
	
	local rewardHistory = {}
	
	NewActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)

		local criticValue = NewActorCriticModel.CriticModel:predict(previousFeatureVector, true)[1][1]

		local numberOfActions = #allOutputsMatrix[1]

		local actionIndex = sampleAction(actionProbabilityVector)

		local action = NewActorCriticModel.ClassesList[actionIndex]

		local actionProbability = actionProbabilityVector[1][actionIndex]

		table.insert(actionProbabilityHistory, actionProbability)

		table.insert(criticValueHistory, criticValue)

		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewActorCriticModel:setEpisodeUpdateFunction(function()
		
		local returnsHistory = {}

		local discountedSum = 0

		local historyLength = #rewardHistory

		for h = historyLength, 1, -1 do

			discountedSum = rewardHistory[h] + NewActorCriticModel.discountFactor * discountedSum

			table.insert(returnsHistory, 1, discountedSum)

		end

		local sumActorLosses = 0

		local sumCriticLosses = 0

		for h = 1, historyLength, 1 do

			local criticValue = criticValueHistory[h]

			local returns = returnsHistory[h]

			local actionProbability = actionProbabilityHistory[h]

			local actorLoss = math.log(actionProbability) * (returns - criticValue) 

			local criticLoss = (returns - criticValue)^2

			sumActorLosses += actorLoss

			sumCriticLosses += criticLoss

		end
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel

		local lossValue = sumActorLosses + sumCriticLosses
		
		local numberOfFeatures, hasBias = ActorModel:getLayer(1)
		
		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local lossVector = AqwamMatrixLibrary:createMatrix(1, #NewActorCriticModel.ClassesList, lossValue)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(lossVector, true)
		CriticModel:backPropagate(lossValue, true)

		table.clear(actionProbabilityHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)
		
	end)
	
	NewActorCriticModel:extendResetFunction(function()
		
		table.clear(actionProbabilityHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)
		
	end)
	
	return NewActorCriticModel
	
end

return ActorCriticModel
