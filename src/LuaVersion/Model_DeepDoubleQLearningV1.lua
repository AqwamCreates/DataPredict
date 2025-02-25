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

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, ReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel:setName("DeepDoubleQLearningV1")
	
	NewDeepDoubleQLearningModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewDeepDoubleQLearningModel.lambda = parameterDictionary.lambda or defaultLambda

	NewDeepDoubleQLearningModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix 
	
	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorVector, temporalDifferenceError = NewDeepDoubleQLearningModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

		Model:forwardPropagate(previousFeatureVector, true, true)
		
		Model:backwardPropagate(temporalDifferenceErrorVector, true)

		NewDeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError
		
	end)
	
	NewDeepDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleQLearningModel.eligibilityTraceMatrix = nil
		
	end)
	
	NewDeepDoubleQLearningModel:setResetFunction(function() 
		
		NewDeepDoubleQLearningModel.eligibilityTraceMatrix = nil
		
	end)

	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleQLearningModel:loadModelParametersFromModelParametersArray(index)
	
	local Model = self.Model
	
	if (not self.ModelParametersArray[1]) and (not self.ModelParametersArray[2]) then
		
		Model:generateLayers()
		
		self:saveModelParametersFromModelParametersArray(1)
		
		self:saveModelParametersFromModelParametersArray(2)
		
	end
	
	local CurrentModelParameters = self.ModelParametersArray[index]
	
	Model:setModelParameters(CurrentModelParameters, true)
	
end

function DeepDoubleQLearningModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
	
	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local lambda = self.lambda
	
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
	
	if (lambda ~= 0) then

		local eligibilityTraceMatrix = self.eligibilityTraceMatrix

		if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

		eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)

		eligibilityTraceMatrix[1][actionIndex] = eligibilityTraceMatrix[1][actionIndex] + 1

		temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTraceMatrix)

		self.eligibilityTraceMatrix = eligibilityTraceMatrix

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