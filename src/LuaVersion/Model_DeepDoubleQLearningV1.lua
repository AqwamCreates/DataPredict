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

DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, DeepReinforcementLearningBaseModel)

function DeepDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearningV1")
	
	NewDeepDoubleQLearningModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}

	NewDeepDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorVector, temporalDifferenceError = NewDeepDoubleQLearningModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.
		
		NewDeepDoubleQLearningModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		Model:forwardPropagate(previousFeatureVector, true)
		
		Model:update(negatedTemporalDifferenceErrorVector, true)

		NewDeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError
		
	end)
	
	NewDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleQLearningModel.EligibilityTrace:reset()
		
	end)
	
	NewDeepDoubleQLearningModel:setResetFunction(function() 
		
		NewDeepDoubleQLearningModel.EligibilityTrace:reset()
		
	end)

	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleQLearningModel:loadModelParametersFromModelParametersArray(index)

	local Model = self.Model

	local ModelParametersArray = self.ModelParametersArray

	if (not ModelParametersArray[index]) then

		Model:generateLayers()

		self:saveModelParametersFromModelParametersArray(index)

	end

	local CurrentModelParameters = ModelParametersArray[index]

	Model:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleQLearningModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
	
	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local EligibilityTrace = self.EligibilityTrace
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
	
	local previousVector = Model:forwardPropagate(previousFeatureVector)
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

	local _, maxQValue = Model:predict(currentFeatureVector)

	local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])
	
	local ClassesList = Model:getClassesList()
	
	local numberOfClasses = #ClassesList

	local actionIndex = table.find(ClassesList, action)
	
	local lastValue = previousVector[1][actionIndex]
	
	local temporalDifferenceError = targetValue - lastValue
	
	local outputDimensionSizeArray = {1, numberOfClasses}
		
	local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

	temporalDifferenceErrorVector[1][actionIndex] = temporalDifferenceError
	
	if (EligibilityTrace) then

		EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)
		
		temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

	end
	
	return temporalDifferenceErrorVector, temporalDifferenceError
	
end

function DeepDoubleQLearningModel:setModelParameters1(ModelParameters1, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.ModelParametersArray[1] = ModelParameters1
		
	else
		
		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)
		
	end

end

function DeepDoubleQLearningModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function DeepDoubleQLearningModel:getModelParameters1(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.ModelParametersArray[1]
		
	else
		
		return self:deepCopyTable(self.ModelParametersArray[1])
		
	end

end

function DeepDoubleQLearningModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepDoubleQLearningModel
