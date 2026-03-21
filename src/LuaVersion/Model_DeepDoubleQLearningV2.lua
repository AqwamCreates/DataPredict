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

local DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, DeepReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, PrimaryModelParameters[layer])

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, TargetModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(PrimaryModelParametersPart, TargetModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearningV2")
	
	NewDeepDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepDoubleQLearningModel.TargetModelParameters = parameterDictionary.TargetModelParameters

	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local discountFactor = NewDeepDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepDoubleQLearningModel.EligibilityTrace
		
		local TargetModelParameters = NewDeepDoubleQLearningModel.TargetModelParameters
		
		local ClassesList = Model:getClassesList()
		
		local PrimaryModelParameters = Model:getModelParameters(true)

		if (not PrimaryModelParameters) then PrimaryModelParameters = Model:generateLayers() end
		
		if (not TargetModelParameters) then TargetModelParameters = PrimaryModelParameters end
		
		local primaryPreviousQVector = Model:forwardPropagate(previousFeatureVector)
		
		local maximumPrimaryCurrentActionVector = Model:predict(currentFeatureVector)
		
		local primaryCurrentActionIndex = table.find(ClassesList, maximumPrimaryCurrentActionVector[1][1])
		
		Model:setModelParameters(TargetModelParameters, true)
		
		local targetCurrentQVector = Model:forwardPropagate(currentFeatureVector)

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * targetCurrentQVector[1][primaryCurrentActionIndex])

		local primaryPreviousActionIndex = table.find(ClassesList, previousAction)

		local primaryPreviousQValue = primaryPreviousQVector[1][primaryPreviousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorVector[1][primaryPreviousActionIndex] = temporalDifferenceError
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(1, primaryPreviousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

		end
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.
		
		Model:setModelParameters(PrimaryModelParameters, true)
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		PrimaryModelParameters = Model:getModelParameters(true)

		NewDeepDoubleQLearningModel.TargetModelParameters = rateAverageModelParameters(NewDeepDoubleQLearningModel.averagingRate, TargetModelParameters, PrimaryModelParameters)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepDoubleQLearningModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)
	
	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:setTargetModelParameters(TargetModelParameters, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.TargetModelParameters = TargetModelParameters

	else

		self.TargetModelParameters = self:deepCopyTable(TargetModelParameters)

	end
	
end

function DeepDoubleQLearningModel:getTargetModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetModelParameters

	else

		return self:deepCopyTable(self.TargetModelParameters)

	end

end

return DeepDoubleQLearningModel
