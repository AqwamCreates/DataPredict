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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local ReinforcementLearningActorCriticBaseModel = require("Model_ReinforcementLearningActorCriticBaseModel")

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, ReinforcementLearningActorCriticBaseModel)

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
	
	local NewActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	NewActorCriticModel:setName("ActorCritic")
	
	local actionProbabilityVectorHistory = {}
	
	local criticValueHistory = {}
	
	local rewardValueHistory = {}
	
	local terminalStateValueHistory = {}
	
	NewActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local CriticModel = NewActorCriticModel.CriticModel

		local actionVector = NewActorCriticModel.ActorModel:forwardPropagate(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local logActionProbabilityVector = AqwamTensorLibrary:logarithm(actionProbabilityVector)
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(rewardValueHistory, rewardValue)
		
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
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, criticValue)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewActorCriticModel.discountFactor)

		local sumCriticLoss = 0
		
		local sumActorLossVector = AqwamTensorLibrary:createTensor({1, #actionProbabilityVectorHistory[1]}, 0)
		
		for h, actionVector in ipairs(actionProbabilityVectorHistory) do
			
			local criticLoss = rewardToGoHistory[h] - criticValueHistory[h]
			
			local actorLossVector = AqwamTensorLibrary:multiply(actionVector, criticLoss)
			
			sumActorLossVector = AqwamTensorLibrary:add(sumActorLossVector, actorLossVector)
			
			sumCriticLoss = sumCriticLoss + criticLoss
			
		end
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel
		
		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 1)
		
		sumActorLossVector = AqwamTensorLibrary:unaryMinus(sumActorLossVector)
		
		ActorModel:forwardPropagate(featureVector, true, true)

		CriticModel:forwardPropagate(featureVector, true, true)
		
		ActorModel:backwardPropagate(sumActorLossVector, true)

		CriticModel:backwardPropagate(sumCriticLoss, true)

		table.clear(actionProbabilityVectorHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)
		
	end)
	
	NewActorCriticModel:setResetFunction(function()
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)
		
	end)
	
	return NewActorCriticModel
	
end

return ActorCriticModel