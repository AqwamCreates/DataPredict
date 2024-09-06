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

	local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(vector)

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
	
	local actionProbabilityVectorHistory = {}
	
	local rewardValueHistory = {}
	
	NewREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewREINFORCEModel.Model

		local actionVector = Model:predict(previousFeatureVector, true)
		
		local actionProbabilityVector = calculateProbability(actionVector)
		
		local logActionProbabilityVector = AqwamMatrixLibrary:logarithm(actionProbabilityVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, expectedActionVector, rewardValue, currentFeatureVector, standardDeviationVector)

		local randomNormalVector = AqwamMatrixLibrary:createRandomNormalMatrix(1, #expectedActionVector[1])

		local actionVectorPart1 = AqwamMatrixLibrary:multiply(standardDeviationVector, randomNormalVector)

		local actionVector = AqwamMatrixLibrary:add(expectedActionVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(actionVector, expectedActionVector)

		local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamMatrixLibrary:logarithm(standardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamMatrixLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamMatrixLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVector = AqwamMatrixLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setEpisodeUpdateFunction(function()
		
		local Model = NewREINFORCEModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewREINFORCEModel.discountFactor)
		
		local sumLossVector = AqwamMatrixLibrary:createMatrix(1, #actionProbabilityVectorHistory[1][1], 0)
		
		for h, actionProbabilityVector in ipairs(actionProbabilityVectorHistory) do
			
			local lossVector = AqwamMatrixLibrary:multiply(actionProbabilityVector, rewardToGoArray[h])

			sumLossVector = AqwamMatrixLibrary:add(sumLossVector, lossVector)
			
		end	
		
		local numberOfFeatures = Model:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		local numberOfActions = #Model:getClassesList()

		sumLossVector = AqwamMatrixLibrary:unaryMinus(sumLossVector)
		
		Model:forwardPropagate(featureVector, true)

		Model:backwardPropagate(sumLossVector, true)
		
		table.clear(actionProbabilityVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewREINFORCEModel:setResetFunction(function()

		table.clear(actionProbabilityVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewREINFORCEModel

end

function REINFORCEModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return REINFORCEModel