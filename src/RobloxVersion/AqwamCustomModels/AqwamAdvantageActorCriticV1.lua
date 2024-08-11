local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.Parent.Models.ReinforcementLearningActorCriticBaseModel)

AqwamAdvantageActorCriticModel = {}

AqwamAdvantageActorCriticModel.__index = AqwamAdvantageActorCriticModel

setmetatable(AqwamAdvantageActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local function sample(vector)

	local totalValue = AqwamMatrixLibrary:sum(vector)
	
	local lowestValue = AqwamMatrixLibrary:findMinimumValueInMatrix(vector)
	
	local highestValue = AqwamMatrixLibrary:findMaximumValueInMatrix(vector)
	
	local randomValue = math.random() - math.random()

	randomValue = math.clamp(randomValue * totalValue, lowestValue, highestValue)
	
	local cumulativeValue = 0

	local vectorIndex = 1

	for i, value in ipairs(vector[1]) do

		cumulativeValue += value

		if (randomValue >= cumulativeValue) then continue end

		vectorIndex = i

		break

	end

	return vectorIndex

end

local function calculateAverage(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

function AqwamAdvantageActorCriticModel.new(discountFactor)

	local NewAqwamAdvantageActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewAqwamAdvantageActorCriticModel, AqwamAdvantageActorCriticModel)

	local advantageHistory = {}

	local averageOutputHistory = {}

	NewAqwamAdvantageActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local allOutputsMatrix = NewAqwamAdvantageActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local averageAllOutputsMatrix = calculateAverage(allOutputsMatrix)

		local CriticModel = NewAqwamAdvantageActorCriticModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewAqwamAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue

		local actionIndex = sample(averageAllOutputsMatrix)

		local outputValue = averageAllOutputsMatrix[1][actionIndex]

		table.insert(advantageHistory, advantageValue)

		table.insert(averageOutputHistory, outputValue)

	end)

	NewAqwamAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local historyLength = #advantageHistory

		local sumActorLosses = 0

		local sumCriticLosses = 0

		for h = 1, historyLength, 1 do

			local advantage = advantageHistory[h]

			local averageOutput = averageOutputHistory[h]

			local actorLoss = averageOutput * advantage

			local criticLoss = math.pow(advantage, 2)

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

		ActorModel:backwardPropagate(actorLossVector, true)
		CriticModel:backwardPropagate(-sumCriticLosses, true)

		table.clear(advantageHistory)

		table.clear(averageOutputHistory)

	end)

	NewAqwamAdvantageActorCriticModel:extendResetFunction(function()

		table.clear(advantageHistory)

		table.clear(averageOutputHistory)

	end)

	return NewAqwamAdvantageActorCriticModel

end

return AqwamAdvantageActorCriticModel
