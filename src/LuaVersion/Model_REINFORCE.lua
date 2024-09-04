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

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

REINFORCEModel = {}

REINFORCEModel.__index = REINFORCEModel

setmetatable(REINFORCEModel, ReinforcementLearningBaseModel)

local function calculateProbability(vector)

	local meanVector = AqwamMatrixLibrary:horizontalMean(vector)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(vector)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(vector, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function REINFORCEModel.new(discountFactor)

	local NewREINFORCEModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewREINFORCEModel, REINFORCEModel)
	
	local actionProbabilityValueHistory = {}
	
	local rewardValueHistory = {}
	
	NewREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewREINFORCEModel.Model

		local actionVector = Model:predict(previousFeatureVector, true)
		
		local actionProbabilityVector = calculateProbability(actionVector)
		
		local actionIndex = table.find(Model:getClassesList(), action)
		
		local actionProbability = actionProbabilityVector[1][actionIndex]
		
		local logActionProbability = math.log(actionProbability)

		table.insert(actionProbabilityValueHistory, logActionProbability)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setCategoricalEpisodeUpdateFunction(function()
		
		local Model = NewREINFORCEModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewREINFORCEModel.discountFactor)
		
		local sumLossValue = 0
		
		for h, actionProbabilityValue in ipairs(actionProbabilityValueHistory) do

			sumLossValue = sumLossValue + (actionProbabilityValue * rewardToGoArray[h])
			
		end	
		
		local numberOfNeurons = Model:getTotalNumberOfNeurons(1)

		local inputVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeurons, 1)
		
		local numberOfLayers = Model:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = Model:getTotalNumberOfNeurons(numberOfLayers)
		
		local sumLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumLossValue)
		
		Model:forwardPropagate(inputVector, true)

		Model:backwardPropagate(sumLossVector, true)
		
		table.clear(actionProbabilityValueHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewREINFORCEModel:setCategoricalResetFunction(function()

		table.clear(actionProbabilityValueHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)
		
		local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:verticalZScoreNormalization(actionVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logStandardDeviationVector = AqwamMatrixLibrary:logarithm(standardDeviationVector)

		local multipliedLogStandardDeviationVector = AqwamMatrixLibrary:multiply(2, logStandardDeviationVector)

		local numberOfActionDimensions = #NewREINFORCEModel.Model:getClassesList()

		local logLikelihoodPart1 = AqwamMatrixLibrary:sum(multipliedLogStandardDeviationVector)

		local logLikelihood = -0.5 * (logLikelihoodPart1 + (numberOfActionDimensions * math.log(2 * math.pi)))
		
		table.insert(actionProbabilityValueHistory, logLikelihood)

		table.insert(rewardValueHistory, rewardValue)
		
	end)
	
	NewREINFORCEModel:setDiagonalGaussianEpisodeUpdateFunction(function()

		local Model = NewREINFORCEModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewREINFORCEModel.discountFactor)

		local sumLossValue = 0

		for h, actionProbabilityValue in ipairs(actionProbabilityValueHistory) do

			sumLossValue = sumLossValue + (actionProbabilityValue * rewardToGoArray[h])

		end	

		local numberOfNeurons = Model:getTotalNumberOfNeurons(1)

		local inputVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeurons, 1)

		local numberOfLayers = Model:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = Model:getTotalNumberOfNeurons(numberOfLayers)

		local sumLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumLossValue)

		Model:forwardPropagate(inputVector, true)

		Model:backwardPropagate(sumLossVector, true)

		table.clear(actionProbabilityValueHistory)

		table.clear(rewardValueHistory)

	end)
	
	NewREINFORCEModel:setDiagonalGaussianResetFunction(function()

		table.clear(actionProbabilityValueHistory)

		table.clear(rewardValueHistory)

	end)
	
	return NewREINFORCEModel

end

function REINFORCEModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return REINFORCEModel