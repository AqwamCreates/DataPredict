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

local DeepReinforcementLearningActorCriticBaseModel = require("Core_DeepReinforcementLearningActorCriticBaseModel")

DeepDeterministicPolicyGradientModel = {}

DeepDeterministicPolicyGradientModel.__index = DeepDeterministicPolicyGradientModel

setmetatable(DeepDeterministicPolicyGradientModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultAveragingRate = 0.995

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDeterministicPolicyGradientModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDeepDeterministicPolicyGradientModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepDeterministicPolicyGradientModel, DeepDeterministicPolicyGradientModel)
	
	NewDeepDeterministicPolicyGradientModel:setName("DeepDeterministicPolicyGradient")
	
	NewDeepDeterministicPolicyGradientModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDeterministicPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)
		
		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end
		
		local ActorModel = NewDeepDeterministicPolicyGradientModel.ActorModel
		
		local CriticModel = NewDeepDeterministicPolicyGradientModel.CriticModel
		
		local averagingRate = NewDeepDeterministicPolicyGradientModel.averagingRate
		
		local currentActionMeanVector = ActorModel:forwardPropagate(currentFeatureVector, true)
		
		local ActorModelParameters = ActorModel:getModelParameters(true)
		
		local targetCriticActionMeanInputVector = AqwamTensorLibrary:concatenate(currentFeatureVector, currentActionMeanVector, 2)
		
		local targetQValue = CriticModel:forwardPropagate(targetCriticActionMeanInputVector, true)[1][1]
		
		local CriticModelParameters = CriticModel:getModelParameters(true)
	
		local yValue = rewardValue + (NewDeepDeterministicPolicyGradientModel.discountFactor * (1 - terminalStateValue) * targetQValue)
		
		local actionVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)
		
		actionVector = AqwamTensorLibrary:add(actionVector, actionMeanVector)
		
		local previousCriticActionInputVector = AqwamTensorLibrary:concatenate(previousFeatureVector, actionVector, 2)
		
		local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputVector, true)[1][1]

		local negatedtemporalDifferenceError = (currentQValue - yValue)
		
		local temporalDifferenceError = -negatedtemporalDifferenceError
		
		ActorModel:forwardPropagate(previousFeatureVector, true)

		ActorModel:update(negatedtemporalDifferenceError, true)
		
		local previousCriticActionMeanInputVector = AqwamTensorLibrary:concatenate(previousFeatureVector, actionMeanVector, 2)
		
		CriticModel:forwardPropagate(previousCriticActionMeanInputVector, true)
		
		CriticModel:update(temporalDifferenceError, true)
		
		local TargetActorModelParameters = ActorModel:getModelParameters(true)
		
		local TargetCriticModelParameters = CriticModel:getModelParameters(true)
		
		TargetActorModelParameters = rateAverageModelParameters(averagingRate, TargetActorModelParameters, ActorModelParameters)
		
		TargetCriticModelParameters = rateAverageModelParameters(averagingRate, TargetCriticModelParameters, CriticModelParameters)
		
		ActorModel:setModelParameters(TargetActorModelParameters, true)
		
		CriticModel:setModelParameters(TargetCriticModelParameters, true)

		return temporalDifferenceError
		
	end)
	
	NewDeepDeterministicPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewDeepDeterministicPolicyGradientModel:setResetFunction(function() end)
	
	return NewDeepDeterministicPolicyGradientModel
	
end

return DeepDeterministicPolicyGradientModel
