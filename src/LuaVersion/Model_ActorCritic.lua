--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningActorCriticBaseModel = require("Model_ReinforcementLearningActorCriticBaseModel")

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
	
	local categoricalActionProbabilityHistory = {}
	
	local categoricalCriticValueHistory = {}
	
	local categoricalRewardHistory = {}
	
	local diagonalGaussianActionProbabilityHistory = {}
	
	local diagonalGaussianCriticValueHistory = {}
	
	local diagonalGaussianRewardHistory = {}
	
	NewActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local ActorModel = NewActorCriticModel.ActorModel
		
		local allOutputsMatrix = ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)

		local criticValue = NewActorCriticModel.CriticModel:predict(previousFeatureVector, true)[1][1]

		local numberOfActions = #allOutputsMatrix[1]

		local actionIndex = table.find(ActorModel:getClassesList(), action)

		local actionProbability = actionProbabilityVector[1][actionIndex]

		table.insert(categoricalActionProbabilityHistory, math.log(actionProbability))

		table.insert(categoricalCriticValueHistory, criticValue)

		table.insert(categoricalRewardHistory, rewardValue)
		
	end)
	
	NewActorCriticModel:setCategoricalEpisodeUpdateFunction(function()
		
		local returnsHistory = {}

		local discountedSum = 0

		local historyLength = #categoricalRewardHistory

		for h = historyLength, 1, -1 do

			discountedSum = categoricalRewardHistory[h] + NewActorCriticModel.discountFactor * discountedSum

			table.insert(returnsHistory, 1, discountedSum)

		end

		local sumActorLoss = 0

		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local criticValue = categoricalCriticValueHistory[h]

			local returnValue = returnsHistory[h]

			local logActionProbability = categoricalActionProbabilityHistory[h]
			
			local criticLoss = returnValue - criticValue

			local actorLoss = logActionProbability * criticLoss

			sumCriticLoss += criticLoss
			
			sumActorLoss += actorLoss

		end
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel
		
		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumActorLoss)

		CriticModel:forwardPropagate(featureVector, true)
		ActorModel:forwardPropagate(featureVector, true)

		CriticModel:backwardPropagate(-sumCriticLoss, true)
		ActorModel:backwardPropagate(sumActorLossVector, true)

		table.clear(categoricalActionProbabilityHistory)

		table.clear(categoricalCriticValueHistory)

		table.clear(categoricalRewardHistory)
		
	end)
	
	NewActorCriticModel:setCategoricalResetFunction(function()
		
		table.clear(categoricalActionProbabilityHistory)

		table.clear(categoricalCriticValueHistory)

		table.clear(categoricalRewardHistory)
		
	end)
	
	NewActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)
		
		local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:verticalZScoreNormalization(actionVector)
		
		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)
		
		local logStandardDeviationVector = AqwamMatrixLibrary:logarithm(standardDeviationVector)
		
		local multipliedLogStandardDeviationVector = AqwamMatrixLibrary:multiply(2, logStandardDeviationVector)
		
		local numberOfActionDimensions = #NewActorCriticModel.ActorModel:getClassesList()
		
		local logLikelihoodPart1 = AqwamMatrixLibrary:sum(multipliedLogStandardDeviationVector)
		
		local logLikelihood = 0.5 * (logLikelihoodPart1 + (numberOfActionDimensions * math.log(2 * math.pi)))

		local criticValue = NewActorCriticModel.CriticModel:predict(previousFeatureVector, true)[1][1]
		
		table.insert(diagonalGaussianActionProbabilityHistory, logLikelihood)

		table.insert(diagonalGaussianCriticValueHistory, criticValue)

		table.insert(diagonalGaussianRewardHistory, rewardValue)
		
	end)
	
	NewActorCriticModel:setDiagonalGaussianEpisodeUpdateFunction(function()

		local returnsHistory = {}

		local discountedSum = 0

		local historyLength = #diagonalGaussianRewardHistory
		
		local discountFactor =  NewActorCriticModel.discountFactor

		for h = historyLength, 1, -1 do

			discountedSum = diagonalGaussianRewardHistory[h] + (discountFactor * discountedSum)

			table.insert(returnsHistory, 1, discountedSum)

		end

		local sumCriticLoss = 0
		
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(#diagonalGaussianActionProbabilityHistory[1], #diagonalGaussianActionProbabilityHistory[1][1], 0)

		for h = 1, historyLength, 1 do

			local criticValue = diagonalGaussianCriticValueHistory[h]

			local returnValue = returnsHistory[h]

			local logActionProbabilityVector = diagonalGaussianActionProbabilityHistory[h]
			
			local criticLoss = returnValue - criticValue

			local actorLossVector = AqwamMatrixLibrary:multiply(logActionProbabilityVector, criticLoss)
			
			sumCriticLoss += criticLoss

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

		end
		
		sumActorLossVector = AqwamMatrixLibrary:multiply(-1, sumActorLossVector)
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		CriticModel:forwardPropagate(featureVector, true)
		ActorModel:forwardPropagate(featureVector, true)

		CriticModel:backwardPropagate(-sumCriticLoss, true)
		ActorModel:backwardPropagate(sumActorLossVector, true)

		table.clear(diagonalGaussianActionProbabilityHistory)

		table.clear(diagonalGaussianCriticValueHistory)

		table.clear(diagonalGaussianRewardHistory)

	end)
	
	NewActorCriticModel:setDiagonalGaussianResetFunction(function()

		table.clear(diagonalGaussianActionProbabilityHistory)

		table.clear(diagonalGaussianCriticValueHistory)

		table.clear(diagonalGaussianRewardHistory)

	end)
	
	return NewActorCriticModel
	
end

return ActorCriticModel