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

local DeepReinforcementLearningActorCriticBaseModel = require(script.Parent.DeepReinforcementLearningActorCriticBaseModel)

local DeepDeterministicPolicyGradientModel = {}

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
	
	NewDeepDeterministicPolicyGradientModel.TargetActorModelParameters = parameterDictionary.TargetActorModelParameters
	
	NewDeepDeterministicPolicyGradientModel.TargetCriticModelParameters = parameterDictionary.TargetCriticModelParameters
	
	NewDeepDeterministicPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)
		
		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end
		
		local ActorModel = NewDeepDeterministicPolicyGradientModel.ActorModel
		
		local CriticModel = NewDeepDeterministicPolicyGradientModel.CriticModel
		
		local averagingRate = NewDeepDeterministicPolicyGradientModel.averagingRate
		
		local TargetActorModelParameters = NewDeepDeterministicPolicyGradientModel.TargetActorModelParameters
		
		local TargetCriticModelParameters = NewDeepDeterministicPolicyGradientModel.TargetCriticModelParameters
		
		local PrimaryActorModelParameters = ActorModel:getModelParameters(true) or ActorModel:generateLayers()
		
		local PrimaryCriticModelParameters = CriticModel:getModelParameters(true) or CriticModel:generateLayers()
		
		TargetActorModelParameters = TargetActorModelParameters or PrimaryActorModelParameters
		
		TargetCriticModelParameters = TargetCriticModelParameters or PrimaryCriticModelParameters
		
		ActorModel:setModelParameters(TargetActorModelParameters)
		
		local targetCurrentActionMeanVector = ActorModel:forwardPropagate(currentFeatureVector)
		
		local targetCriticActionMeanInputVector = AqwamTensorLibrary:concatenate(currentFeatureVector, targetCurrentActionMeanVector, 2)
		
		CriticModel:setModelParameters(TargetCriticModelParameters)
		
		local targetQValue = CriticModel:forwardPropagate(targetCriticActionMeanInputVector)[1][1]
		
		local CriticModelParameters = CriticModel:getModelParameters(true)
	
		local yValue = rewardValue + (NewDeepDeterministicPolicyGradientModel.discountFactor * (1 - terminalStateValue) * targetQValue)
		
		ActorModel:setModelParameters(PrimaryActorModelParameters)

		ActorModel:forwardPropagate(previousFeatureVector, true)
		
		local previousCriticActionInputVector = AqwamTensorLibrary:concatenate(previousFeatureVector, previousActionMeanVector, 2)
		
		CriticModel:setModelParameters(PrimaryCriticModelParameters)
		
		local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputVector, true)[1][1]

		local criticError  = 2 * (currentQValue - yValue)
		
		local temporalDifferenceError = -criticError
		
		ActorModel:update(-currentQValue, true)
		
		CriticModel:update(temporalDifferenceError, true)
		
		PrimaryActorModelParameters = ActorModel:getModelParameters(true)
		
		PrimaryCriticModelParameters = CriticModel:getModelParameters(true)
		
		NewDeepDeterministicPolicyGradientModel.TargetActorModelParameters = rateAverageModelParameters(averagingRate, TargetActorModelParameters, PrimaryActorModelParameters)
		
		NewDeepDeterministicPolicyGradientModel.TargetCriticModelParameters = rateAverageModelParameters(averagingRate, TargetCriticModelParameters, PrimaryCriticModelParameters)

		return temporalDifferenceError
		
	end)
	
	NewDeepDeterministicPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue) end)
	
	NewDeepDeterministicPolicyGradientModel:setResetFunction(function() end)
	
	return NewDeepDeterministicPolicyGradientModel
	
end

function DeepDeterministicPolicyGradientModel:setTargetActorModelParameters(TargetActorModelParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetActorModelParameters = TargetActorModelParameters

	else

		self.TargetActorModelParameters = self:deepCopyTable(TargetActorModelParameters)

	end

end

function DeepDeterministicPolicyGradientModel:setTargetCriticModelParameters(TargetCriticModelParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticModelParameters = TargetCriticModelParameters

	else

		self.TargetCriticModelParameters = self:deepCopyTable(TargetCriticModelParameters)

	end

end

function DeepDeterministicPolicyGradientModel:getTargetActorModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetActorModelParameters

	else

		return self:deepCopyTable(self.TargetActorModelParameters)

	end

end

function DeepDeterministicPolicyGradientModel:getTargetCriticModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticModelParameters

	else

		return self:deepCopyTable(self.TargetCriticModelParameters)

	end

end

return DeepDeterministicPolicyGradientModel
