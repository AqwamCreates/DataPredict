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

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

TwinDelayedDeepDeterministicPolicyGradientModel = {}

TwinDelayedDeepDeterministicPolicyGradientModel.__index = TwinDelayedDeepDeterministicPolicyGradientModel

setmetatable(TwinDelayedDeepDeterministicPolicyGradientModel, ReinforcementLearningActorCriticBaseModel)

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
	
	local NewTwinDelayedDeepDeterministicPolicyGradient = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	setmetatable(NewTwinDelayedDeepDeterministicPolicyGradient, TwinDelayedDeepDeterministicPolicyGradientModel)
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setName("TwinDelayedDeepDeterministicPolicyGradient")
	
	NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor = parameterDictionary.noiseClippingFactor or defaultNoiseClippingFactor
	
	NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount = parameterDictionary.policyDelayAmount or defaultPolicyDelayAmount
	
	NewTwinDelayedDeepDeterministicPolicyGradient.CriticModelParametersArray = parameterDictionary.CriticModelParametersArray or {}
	
	local TargetCriticModelParametersArray = {}
	
	local currentNumberOfUpdate = 0
	
	NewTwinDelayedDeepDeterministicPolicyGradient:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)
		
		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end
		
		local ActorModel = NewTwinDelayedDeepDeterministicPolicyGradient.ActorModel
		
		local CriticModel = NewTwinDelayedDeepDeterministicPolicyGradient.CriticModel
		
		local averagingRate = NewTwinDelayedDeepDeterministicPolicyGradient.averagingRate
		
		local noiseClippingFactor = NewTwinDelayedDeepDeterministicPolicyGradient.noiseClippingFactor
		
		local CriticModelParametersArray = NewTwinDelayedDeepDeterministicPolicyGradient.CriticModelParametersArray
		
		local noiseClipFunction = function(value) return math.clamp(value, -noiseClippingFactor, noiseClippingFactor) end
		
		local currentActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]})
		
		local clippedCurrentActionNoiseVector = AqwamTensorLibrary:applyFunction(noiseClipFunction, currentActionNoiseVector)
		
		local previousActionVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

		previousActionVector = AqwamTensorLibrary:add(previousActionVector, actionMeanVector)

		local previousActionArray = previousActionVector[1] 

		local lowestActionValue = math.min(table.unpack(previousActionArray))

		local highestActionValue = math.max(table.unpack(previousActionArray))
		
		local currentActionMeanVector = ActorModel:forwardPropagate(currentFeatureVector, true)
		
		local ActorModelParameters = ActorModel:getModelParameters(true)
		
		local targetActionVectorPart1 = AqwamTensorLibrary:add(currentActionMeanVector, clippedCurrentActionNoiseVector)
		
		local actionClipFunction = function(value)
			
			if (lowestActionValue ~= lowestActionValue) or (highestActionValue ~= highestActionValue) then
				
				error("Received Nan values.")
			
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

			local CriticModelParameters = CriticModel:getModelParameters(true)

			TargetCriticModelParametersArray[i] = CriticModelParameters

		end

		local minimumCurrentCriticValue = math.min(table.unpack(currentCriticValueArray))
		
		local yValuePart1 = NewTwinDelayedDeepDeterministicPolicyGradient.discountFactor * (1 - terminalStateValue) * minimumCurrentCriticValue
		
		local yValue = rewardValue + yValuePart1
		
		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor({1, 2}, 0)

		local previousCriticValueArray = {}
		
		local previousCriticActionMeanInputVector = AqwamTensorLibrary:concatenate(previousFeatureVector, actionMeanVector, 2)
		
		for i = 1, 2, 1 do 

			CriticModel:setModelParameters(CriticModelParametersArray[i], true)

			local previousCriticValue = CriticModel:forwardPropagate(previousCriticActionMeanInputVector, true)[1][1] 

			local criticLoss = yValue - previousCriticValue

			temporalDifferenceErrorVector[1][i] = criticLoss

			previousCriticValueArray[i] = previousCriticValue

			CriticModel:backwardPropagate(criticLoss, true)

			CriticModelParametersArray[i] = CriticModel:getModelParameters(true)

		end
		
		currentNumberOfUpdate = currentNumberOfUpdate + 1
		
		if ((currentNumberOfUpdate % NewTwinDelayedDeepDeterministicPolicyGradient.policyDelayAmount) == 0) then
			
			local actionVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

			actionVector = AqwamTensorLibrary:add(actionVector, actionMeanVector)

			local previousCriticActionInputVector = AqwamTensorLibrary:concatenate(previousFeatureVector, actionVector, 2)
			
			CriticModel:setModelParameters(CriticModelParametersArray[1], true)

			local currentQValue = CriticModel:forwardPropagate(previousCriticActionInputVector, true)[1][1]

			ActorModel:forwardPropagate(previousFeatureVector, true)

			ActorModel:backwardPropagate(-currentQValue, true)

			for i = 1, 2, 1 do TargetCriticModelParametersArray[i] = rateAverageModelParameters(averagingRate, TargetCriticModelParametersArray[i], CriticModelParametersArray[i]) end
			
			local TargetActorModelParameters = ActorModel:getModelParameters(true)

			TargetActorModelParameters = rateAverageModelParameters(averagingRate, TargetActorModelParameters, ActorModelParameters)

			ActorModel:setModelParameters(TargetActorModelParameters, true)
			
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

function TwinDelayedDeepDeterministicPolicyGradientModel:setCrtiticModelParameters1(CriticModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticModelParametersArray[1] = CriticModelParameters1

	else

		self.CriticModelParametersArray[1] = self:deepCopyTable(CriticModelParameters1)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:setCriticModelParameters2(CriticModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.CriticModelParametersArray[2] = CriticModelParameters2

	else

		self.CriticModelParametersArray[2] = self:deepCopyTable(CriticModelParameters2)

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getCriticModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticModelParametersArray[1]

	else

		return self:deepCopyTable(self.CriticModelParametersArray[1])

	end

end

function TwinDelayedDeepDeterministicPolicyGradientModel:getCriticModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.CriticModelParametersArray[2]

	else

		return self:deepCopyTable(self.CriticModelParametersArray[2])

	end

end

return TwinDelayedDeepDeterministicPolicyGradientModel