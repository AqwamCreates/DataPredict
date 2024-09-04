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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

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
	
	local actionProbabilityValueHistory = {}
	
	local rewardValueHistory = {}
	
	NewREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewREINFORCEModel.Model

		local actionVector = Model:predict(previousFeatureVector, true)
		
		local actionProbabilityVector = calculateProbability(actionVector)
		
		local actionIndex = table.find(Model:getClassesList(), action)
		
		local actionProbabilityValue = actionProbabilityVector[1][actionIndex]
		
		local logActionProbabilityValue = math.log(actionProbabilityValue)

		table.insert(actionProbabilityValueHistory, logActionProbabilityValue)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

		local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(actionVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logStandardDeviationVector = AqwamMatrixLibrary:logarithm(standardDeviationVector)

		local multipliedLogStandardDeviationVector = AqwamMatrixLibrary:multiply(2, logStandardDeviationVector)

		local numberOfActionDimensions = #NewREINFORCEModel.Model:getClassesList()

		local actionProbabilityValuePart1 = AqwamMatrixLibrary:sum(multipliedLogStandardDeviationVector)

		local actionProbabilityValue = -0.5 * (actionProbabilityValuePart1 + (numberOfActionDimensions * math.log(2 * math.pi)))

		table.insert(actionProbabilityValueHistory, actionProbabilityValue)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setEpisodeUpdateFunction(function()
		
		local Model = NewREINFORCEModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewREINFORCEModel.discountFactor)
		
		local sumLossValue = 0
		
		for h, actionProbabilityValue in ipairs(actionProbabilityValueHistory) do

			sumLossValue = sumLossValue + (actionProbabilityValue * rewardToGoArray[h])
			
		end	
		
		local numberOfFeatures = Model:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		local numberOfActions = #Model:getClassesList()

		local sumLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, -sumLossValue)
		
		Model:forwardPropagate(featureVector, true)

		Model:backwardPropagate(sumLossVector, true)
		
		table.clear(actionProbabilityValueHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewREINFORCEModel:setResetFunction(function()

		table.clear(actionProbabilityValueHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewREINFORCEModel

end

function REINFORCEModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return REINFORCEModel