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

local TwinDelayedDeepDeterministicPolicyGradientModel = {}

TwinDelayedDeepDeterministicPolicyGradientModel.__index = TwinDelayedDeepDeterministicPolicyGradientModel

setmetatable(TwinDelayedDeepDeterministicPolicyGradientModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultAveragingRate = 0.995

local defaultNoiseClippingFactor = 0.5

local defaultPolicyDelayAmount = 3

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

function TwinDelayedDeepDeterministicPolicyGradientModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTwinDelayedDeepDeterministicPolicyGradient = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewTwinDelayedDeepDeterministicPolicyGradient, TwinDelayedDeepDeterministicPolicyGradientModel)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setName("TwinDelayedDeepDeterministicPolicyGradient")
	
	NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor = parameterDictionary.noiseClippingFactor or defaultNoiseClippingFactor

	NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount = parameterDictionary.policyDelayAmount or defaultPolicyDelayAmount
	
	NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTwinDelayedDeepDeterministicPolicyGradient.TargetActorModelParameters = parameterDictionary.TargetActorModelParameters
	
	NewTwinDelayedDeepDeterministicPolicyGradient.PrimaryCriticModelParametersArray = parameterDictionary.PrimaryCriticModelParametersArray or {}
	
	NewTwinDelayedDeepDeterministicPolicyGradient.TargetCriticModelParametersArray = parameterDictionary.TargetCriticModelParametersArray or {}
	
	local currentNumberOfUpdate = 0
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)
		
		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end
		
		local ActorModel = NewTwinDelayedDeepDeterministicPolicyGradient.ActorModel
		
		local CriticModel = NewTwinDelayedDeepDeterministicPolicyGradient.CriticModel
		
		local averagingRate = NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate
		
		local noiseClippingFactor = NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor
		
		local TargetActorModelParameters = NewTwinDelayedDeepDeterministicPolicyGradient.TargetActorModelParameters
		
		local PrimaryCriticModelParametersArray = NewTwinDelayedDeepDeterministicPolicyGradient.PrimaryCriticModelParametersArray
		
		local TargetCriticModelParametersArray = NewTwinDelayedDeepDeterministicPolicyGradient.TargetCriticModelParametersArray
		
		local PrimaryActorModelParameters = ActorModel:getModelParameters(true) or ActorModel:generateLayers()
		
		PrimaryCriticModelParametersArray[1] = CriticModel:getModelParameters(true) or CriticModel:generateLayers()
		
		PrimaryCriticModelParametersArray[2] = CriticModel:getModelParameters(true) or CriticModel:generateLayers()
		
		TargetActorModelParameters = TargetActorModelParameters or PrimaryActorModelParameters
		
		TargetCriticModelParametersArray[1] = TargetCriticModelParametersArray[1] or PrimaryCriticModelParametersArray[1]
		
		TargetCriticModelParametersArray[2] = TargetCriticModelParametersArray[2] or PrimaryCriticModelParametersArray[2]
		
		local noiseClipFunction = function(value) return math.clamp(value, -noiseClippingFactor, noiseClippingFactor) end
		
		local currentActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]})
		
		local clippedCurrentActionNoiseVector = AqwamTensorLibrary:applyFunction(noiseClipFunction, currentActionNoiseVector)
		
		local previousActionVector = AqwamTensorLibrary:multiply(previousActionStandardDeviationVector, previousActionNoiseVector)

		previousActionVector = AqwamTensorLibrary:add(previousActionVector, previousActionMeanVector)

		local previousActionArray = previousActionVector[1] 

		local lowestActionValue = math.min(table.unpack(previousActionArray))

		local highestActionValue = math.max(table.unpack(previousActionArray))
		
		local ActorModelParameters = ActorModel:getModelParameters(true)
		
		ActorModel:setModelParameters(TargetActorModelParameters)
		
		local targetCurrentActionMeanVector = ActorModel:forwardPropagate(currentFeatureVector)
		
		local targetActionVectorPart1 = AqwamTensorLibrary:add(targetCurrentActionMeanVector, clippedCurrentActionNoiseVector)
		
		local actionClipFunction = function(value)
			
			if (lowestActionValue ~= lowestActionValue) or (highestActionValue ~= highestActionValue) then
				
				error("Received nan values.")
			
			elseif (lowestActionValue < highestActionValue) then
				
				return math.clamp(value, lowestActionValue, highestActionValue) 
				
			elseif (lowestActionValue > highestActionValue) then
				
				return math.clamp(value, highestActionValue, lowestActionValue)
				
			else
				
				return lowestActionValue
				
			end
			
		end
		
		local targetActionVector = AqwamTensorLibrary:applyFunction(actionClipFunction, targetActionVectorPart1)
		
		local targetCriticActionInputVector = AqwamTensorLibrary:concatenate(currentFeatureVector, targetActionVector, 2)
		
		local currentCriticValueArray = {}
		
		for i = 1, 2, 1 do 

			CriticModel:setModelParameters(TargetCriticModelParametersArray[i])

			currentCriticValueArray[i] = CriticModel:forwardPropagate(targetCriticActionInputVector)[1][1] 

		end

		local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))
		
		local yValuePart1 = NewTwinDelayedDeepDeterministicPolicyGradient.discountFactor * (1 - terminalStateValue) * minimumCurrentCriticValue
		
		local yValue = rewardValue + yValuePart1
		
		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor({1, 2}, 0)

		local previousCriticActionMeanInputVector = AqwamTensorLibrary:concatenate(previousFeatureVector, previousActionMeanVector, 2)
		
		for i = 1, 2, 1 do

			CriticModel:setModelParameters(PrimaryCriticModelParametersArray[i], true)

			local previousCriticValue = CriticModel:forwardPropagate(previousCriticActionMeanInputVector, true)[1][1] 

			local criticLoss = 2 * (previousCriticValue - yValue)

			temporalDifferenceErrorVector[1][i] = -criticLoss -- We perform gradient descent here, so the critic loss is negated so that it can be used as temporal difference value.

			CriticModel:update(criticLoss, true)
			
			PrimaryCriticModelParametersArray[i] = CriticModel:getModelParameters(true)

		end
		
		currentNumberOfUpdate = currentNumberOfUpdate + 1
		
		if ((currentNumberOfUpdate % NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount) == 0) then
			
			CriticModel:setModelParameters(PrimaryCriticModelParametersArray[1], true)

			local currentQValue = CriticModel:forwardPropagate(previousCriticActionMeanInputVector, true)[1][1]
			
			ActorModel:setModelParameters(PrimaryActorModelParameters, true)

			ActorModel:forwardPropagate(previousFeatureVector, true)

			ActorModel:update(-currentQValue, true)

			for i = 1, 2, 1 do TargetCriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetCriticModelParametersArray[i], PrimaryCriticModelParametersArray[i]) end
			
			local PrimaryActorModelParameters = ActorModel:getModelParameters(true)

			NewTwinDelayedDeepDeterministicPolicyGradient.TargetActorModelParameters = rateAverageModelParameters(averagingRate, TargetActorModelParameters, PrimaryActorModelParameters)
			
		end

		return temporalDifferenceErrorVector
		
	end)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setEpisodeUpdateFunction(function() 
		
		currentNumberOfUpdate = 0
		
	end)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setResetFunction(function() 
		
		currentNumberOfUpdate = 0
		
	end)
	
	return NewTwinDelayedDeepDeterministicPolicyGradient
	
end

function TwinDelayedDeepDeterministicPolicyGradientModel:setTargetActorModelParameters(TargetActorModelParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetActorModelParameters = TargetActorModelParameters

	else

		self.TargetActorModelParameters = self:deepCopyTable(TargetActorModelParameters)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setPrimaryCrtiticModelParameters1(PrimaryCriticModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticModelParametersArray[1] = PrimaryCriticModelParameters1

	else

		self.PrimaryCriticModelParametersArray[1] = self:deepCopyTable(PrimaryCriticModelParameters1)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setPrimaryCriticModelParameters2(PrimaryCriticModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.PrimaryCriticModelParametersArray[2] = PrimaryCriticModelParameters2

	else

		self.PrimaryCriticModelParametersArray[2] = self:deepCopyTable(PrimaryCriticModelParameters2)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setTargetCrtiticModelParameters1(TargetCriticModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticModelParametersArray[1] = TargetCriticModelParameters1

	else

		self.TargetCriticModelParametersArray[1] = self:deepCopyTable(TargetCriticModelParameters1)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setTargetCriticModelParameters2(TargetCriticModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetCriticModelParametersArray[2] = TargetCriticModelParameters2

	else

		self.TargetCriticModelParametersArray[2] = self:deepCopyTable(TargetCriticModelParameters2)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getTargetActorModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetActorModelParameters

	else

		return self:deepCopyTable(self.TargetActorModelParameters)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getPrimaryCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticModelParametersArray[1]

	else

		return self:deepCopyTable(self.PrimaryCriticModelParametersArray[1])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getPrimaryCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.PrimaryCriticModelParametersArray[2]

	else

		return self:deepCopyTable(self.PrimaryCriticModelParametersArray[2])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getTargetCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticModelParametersArray

	else

		return self:deepCopyTable(self.TargetCriticModelParametersArray[1])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getTargetCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetCriticModelParametersArray

	else

		return self:deepCopyTable(self.TargetCriticModelParametersArray[2])

	end

end

return TwinDelayedDeepDeterministicPolicyGradientModel
