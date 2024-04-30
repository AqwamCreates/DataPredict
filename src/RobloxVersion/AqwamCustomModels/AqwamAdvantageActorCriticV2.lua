local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.Parent.Models.ReinforcementLearningActorCriticBaseModel)

AqwamAdvantageActorCriticModel = {}

AqwamAdvantageActorCriticModel.__index = AqwamAdvantageActorCriticModel

setmetatable(AqwamAdvantageActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local function sample(probabilityVector)

	local totalProbability = 0

	for _, probability in ipairs(probabilityVector[1]) do

		totalProbability += probability

	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0

	local index = 1

	for i, probability in ipairs(probabilityVector[1]) do

		cumulativeProbability += probability

		if (randomValue > cumulativeProbability) then continue end

		index = i

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

function AqwamAdvantageActorCriticModel.new(discountFactor)

	local NewAqwamAdvantageActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewAqwamAdvantageActorCriticModel, AqwamAdvantageActorCriticModel)

	local advantageHistory = {}

	local probabilityHistory = {}

	NewAqwamAdvantageActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local allOutputsMatrix = NewAqwamAdvantageActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local probabilityMatrix = calculateProbability(allOutputsMatrix)

		local CriticModel = NewAqwamAdvantageActorCriticModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewAqwamAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue

		local actionIndex = sample(probabilityMatrix)

		local probability = probabilityMatrix[1][actionIndex]

		table.insert(advantageHistory, advantageValue)

		table.insert(probabilityHistory, probability)

	end)

	NewAqwamAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local historyLength = #advantageHistory

		local sumActorLosses = 0

		local sumCriticLosses = 0

		for h = 1, historyLength, 1 do

			local advantage = advantageHistory[h]

			local probability = probabilityHistory[h]

			local actorLoss = math.log(probability) * advantage

			local criticLoss = advantage

			sumActorLosses += actorLoss

			sumCriticLosses += criticLoss

		end

		local ActorModel = NewAqwamAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAqwamAdvantageActorCriticModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)
		
		local numberOfLayers = ActorModel:getNumberOfLayers()
		
		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		
		local actorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumActorLosses)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(actorLossVector, true)
		CriticModel:backPropagate(-sumCriticLosses, true)

		table.clear(advantageHistory)

		table.clear(probabilityHistory)

	end)

	NewAqwamAdvantageActorCriticModel:extendResetFunction(function()

		table.clear(advantageHistory)

		table.clear(probabilityHistory)

	end)

	return NewAqwamAdvantageActorCriticModel

end

return AqwamAdvantageActorCriticModel
