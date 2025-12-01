--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local DeepReinforcementLearningBaseModel = require("Model_DeepReinforcementLearningBaseModel")

local REINFORCEModel = {}

REINFORCEModel.__index = REINFORCEModel

setmetatable(REINFORCEModel, DeepReinforcementLearningBaseModel)

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

	local NewREINFORCEModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewREINFORCEModel, REINFORCEModel)

	NewREINFORCEModel:setName("REINFORCE")

	local featureVectorArray = {}

	local actionProbabilityGradientVectorHistory = {}

	local rewardValueHistory = {}

	NewREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

		local Model = NewREINFORCEModel.Model

		local actionVector = Model:forwardPropagate(previousFeatureVector)

		local actionProbabilityVector = calculateProbability(actionVector)

		local ClassesList = Model:getClassesList()
		
		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientVector[i] = (((i == classIndex) and 1) or 0) - actionProbabilityVector[1][i]

		end

		actionProbabilityGradientVector = {actionProbabilityGradientVector}

		table.insert(featureVectorArray, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end

		local actionVectorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationVector, previousActionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(previousActionMeanVector, actionVectorPart1)

		local actionProbabilityGradientVectorPart1 = AqwamTensorLibrary:subtract(actionVector, previousActionMeanVector)

		local actionProbabilityGradientVectorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationVector, 2)

		local actionProbabilityGradientVector = AqwamTensorLibrary:divide(actionProbabilityGradientVectorPart1, actionProbabilityGradientVectorPart2)

		table.insert(featureVectorArray, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewREINFORCEModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewREINFORCEModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewREINFORCEModel.discountFactor)

		for h, actionProbabilityGradientVector in ipairs(actionProbabilityGradientVectorHistory) do

			local lossVector = AqwamTensorLibrary:multiply(actionProbabilityGradientVector, rewardToGoArray[h])

			lossVector = AqwamTensorLibrary:unaryMinus(lossVector)

			Model:forwardPropagate(featureVectorArray[h], true)

			Model:update(lossVector, true)

		end

		table.clear(featureVectorArray)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

	end)

	NewREINFORCEModel:setResetFunction(function()

		table.clear(featureVectorArray)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewREINFORCEModel

end

return REINFORCEModel
