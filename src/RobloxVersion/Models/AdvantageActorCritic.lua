--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, ReinforcementLearningActorCriticBaseModel)

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

function ActorCriticModel.new(discountFactor)
	
	local NewActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	local actionProbabilityHistory = {}
	
	local criticValueHistory = {}
	
	local rewardHistory = {}
	
	NewActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)

		local criticValue = NewActorCriticModel.CriticModel:predict(previousFeatureVector, true)[1][1]

		local numberOfActions = #allOutputsMatrix[1]

		local actionIndex = table.find(NewActorCriticModel.ActorModel:getClassesList(), action)

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

			local returnValue = returnsHistory[h]

			local actionProbability = actionProbabilityHistory[h]

			local actorLoss = math.log(actionProbability) * (returnValue - criticValue) 

			local criticLoss = returnValue - criticValue

			sumActorLosses += actorLoss

			sumCriticLosses += criticLoss

		end
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel
		
		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local actorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumActorLosses)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backwardPropagate(actorLossVector, true)
		CriticModel:backwardPropagate(-sumCriticLosses, true)

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