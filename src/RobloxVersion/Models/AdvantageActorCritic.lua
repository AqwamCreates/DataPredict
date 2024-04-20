local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

setmetatable(AdvantageActorCriticModel, ReinforcementLearningActorCriticBaseModel)

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

	local meanVector = AqwamMatrixLibrary:horizontalMean(outputMatrix)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(outputMatrix)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(outputMatrix, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local zScoreSquaredVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, zScoreSquaredVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:applyFunction(math.exp, probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

function AdvantageActorCriticModel.new(discountFactor)

	local NewAdvantageActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)
	
	local advantageHistory = {}

	local actionProbabilityHistory = {}
	
	NewAdvantageActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewAdvantageActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue

		local numberOfActions = #allOutputsMatrix[1]

		local actionIndex
		
		if action then

			actionIndex = table.find(NewAdvantageActorCriticModel.ActorModel:getClassesList(), action)

		else

			actionIndex = sampleAction(actionProbabilityVector)

		end

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

		ActorModel:backPropagate(-sumActorLosses, true)
		CriticModel:backPropagate(-sumCriticLosses, true)

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
