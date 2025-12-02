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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local DeepReinforcementLearningBaseModel = require(script.Parent.DeepReinforcementLearningBaseModel)

local DeepREINFORCEModel = {}

DeepREINFORCEModel.__index = DeepREINFORCEModel

setmetatable(DeepREINFORCEModel, DeepReinforcementLearningBaseModel)

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

function DeepREINFORCEModel.new(parameterDictionary)

	local NewDeepREINFORCEModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepREINFORCEModel, DeepREINFORCEModel)

	NewDeepREINFORCEModel:setName("DeepREINFORCE")

	local featureVectorArray = {}

	local actionProbabilityGradientVectorHistory = {}

	local rewardValueHistory = {}

	NewDeepREINFORCEModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

		local Model = NewDeepREINFORCEModel.Model

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

	NewDeepREINFORCEModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

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

	NewDeepREINFORCEModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewDeepREINFORCEModel.Model

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewDeepREINFORCEModel.discountFactor)

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

	NewDeepREINFORCEModel:setResetFunction(function()

		table.clear(featureVectorArray)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewDeepREINFORCEModel

end

return DeepREINFORCEModel
