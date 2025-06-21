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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

REINFORCEModel = {}

REINFORCEModel.__index = REINFORCEModel

setmetatable(REINFORCEModel, ReinforcementLearningBaseModel)

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

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

function REINFORCEModel.new(parameterDictionary)

	local NewREINFORCEModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewREINFORCEModel, REINFORCEModel)
	
	NewREINFORCEModel:setName("REINFORCE")
	
	local featureVectorArray = {}
	
	local actionProbabilityVectorHistory = {}
	
	local rewardValueHistory = {}
	
	NewREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

		local actionVector = NewREINFORCEModel.Model:forwardPropagate(previousFeatureVector)
		
		local actionProbabilityVector = calculateProbability(actionVector)
		
		local logActionProbabilityVector = AqwamTensorLibrary:logarithm(actionProbabilityVector)
		
		table.insert(featureVectorArray, previousFeatureVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)
		
		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end

		local actionVectorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(actionMeanVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamTensorLibrary:subtract(actionVector, actionMeanVector)

		local zScoreVector = AqwamTensorLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

		local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamTensorLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVectorPart4 = AqwamTensorLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))
		
		local logActionProbabilityVector = AqwamTensorLibrary:multiply(-0.5, logActionProbabilityVectorPart4)
		
		table.insert(featureVectorArray, previousFeatureVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewREINFORCEModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local Model = NewREINFORCEModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewREINFORCEModel.discountFactor)
		
		for h, actionProbabilityVector in ipairs(actionProbabilityVectorHistory) do
			
			local lossVector = AqwamTensorLibrary:multiply(actionProbabilityVector, rewardToGoArray[h])
			
			lossVector = AqwamTensorLibrary:unaryMinus(lossVector)
			
			Model:forwardPropagate(featureVectorArray[h], true)

			Model:update(lossVector, true)
			
		end
		
		table.clear(featureVectorArray)

		table.clear(actionProbabilityVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewREINFORCEModel:setResetFunction(function()
		
		table.clear(featureVectorArray)

		table.clear(actionProbabilityVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewREINFORCEModel

end

return REINFORCEModel