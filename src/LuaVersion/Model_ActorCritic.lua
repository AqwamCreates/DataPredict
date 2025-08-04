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

local DeepReinforcementLearningActorCriticBaseModel = require("Model_DeepReinforcementLearningActorCriticBaseModel")

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, DeepReinforcementLearningActorCriticBaseModel)

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function ActorCriticModel.new(parameterDictionary)

	local NewActorCriticModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewActorCriticModel, ActorCriticModel)

	NewActorCriticModel:setName("ActorCritic")

	local featureVectorHistory = {}

	local actionProbabilityVectorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	NewActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

		local actionVector = NewActorCriticModel.ActorModel:forwardPropagate(previousFeatureVector, true)

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)

		local logActionProbabilityVector = AqwamTensorLibrary:logarithm(actionProbabilityVector)

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

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

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel

		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewActorCriticModel.discountFactor)

		for h, featureVector in ipairs(featureVectorHistory) do

			local criticLoss = rewardToGoHistory[h] - criticValueHistory[h]

			local actorLossVector = AqwamTensorLibrary:multiply(actionProbabilityVectorHistory[h], criticLoss)

			actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:update(criticLoss, true)

			ActorModel:update(actorLossVector, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	NewActorCriticModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	return NewActorCriticModel

end

return ActorCriticModel
