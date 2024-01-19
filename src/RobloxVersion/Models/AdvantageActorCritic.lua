local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningActorCriticNeuralNetworkBaseModel)

AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

setmetatable(AdvantageActorCriticModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)

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

function AdvantageActorCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewAdvantageActorCriticModel = ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)
	
	local advantageHistory = {}

	local actionProbabilityHistory = {}
	
	NewAdvantageActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewAdvantageActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (currentCriticValue - previousCriticValue))

		local numberOfActions = #allOutputsMatrix[1]

		local actionIndex = sampleAction(actionProbabilityVector)

		local action = NewAdvantageActorCriticModel.ClassesList[actionIndex]

		local actionProbability = actionProbabilityVector[1][actionIndex]

		table.insert(advantageHistory, advantageValue)

		table.insert(actionProbabilityHistory, actionProbability)

	end)

	NewAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local historyLength = #advantageHistory

		local sumActorLosses = 0

		local sumCriticLosses = 0

		for h = 1, historyLength, 1 do

			local advantage = advantageHistory[h]

			local actionProbability = actionProbabilityHistory[h]

			local actorLoss = math.log(actionProbability) * advantage

			local criticLoss = math.pow(advantage, 2)

			sumActorLosses += actorLoss

			sumCriticLosses += criticLoss

		end
		
		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel
		
		local numberOfFeatures, hasBias = ActorModel:getLayer(1)

		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(sumActorLosses, true)
		CriticModel:backPropagate(sumCriticLosses, true)

		table.clear(advantageHistory)

		table.clear(actionProbabilityHistory)

	end)

	NewAdvantageActorCriticModel:extendResetFunction(function()

		table.clear(advantageHistory)

		table.clear(actionProbabilityHistory)

	end)

	return NewAdvantageActorCriticModel

end

return AdvantageActorCriticModel
