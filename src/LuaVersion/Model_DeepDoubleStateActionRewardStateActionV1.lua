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

DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)
	
	NewDeepDoubleStateActionRewardStateActionModel:setName("DeepDoubleStateActionRewardStateActionV1")

	NewDeepDoubleStateActionRewardStateActionModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewDeepDoubleStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda

	NewDeepDoubleStateActionRewardStateActionModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix

	NewDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorVector = NewDeepDoubleStateActionRewardStateActionModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

		Model:forwardPropagate(previousFeatureVector, true, true)
		
		Model:backwardPropagate(temporalDifferenceErrorVector, true)

		NewDeepDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	NewDeepDoubleStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewDeepDoubleStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

function DeepDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)
	
	local Model = self.Model

	if (not self.ModelParametersArray[1]) and (not self.ModelParametersArray[2]) then

		Model:generateLayers()

		self:saveModelParametersFromModelParametersArray(1)

		self:saveModelParametersFromModelParametersArray(2)

	end

	local CurrentModelParameters = self.ModelParametersArray[index]

	Model:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleStateActionRewardStateActionModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
	
	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local lambda = self.lambda
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
	
	local previousVector = Model:forwardPropagate(previousFeatureVector)
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

	local qVector = Model:forwardPropagate(currentFeatureVector)

	local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, qVector, (1 - terminalStateValue))

	local targetVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)
	
	local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetVector, previousVector)
	
	if (lambda ~= 0) then

		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		local eligibilityTraceMatrix = self.eligibilityTraceMatrix

		if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor({1, #ClassesList}, 0) end

		eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)

		eligibilityTraceMatrix[1][actionIndex] = eligibilityTraceMatrix[1][actionIndex] + 1

		temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTraceMatrix)

		self.eligibilityTraceMatrix = eligibilityTraceMatrix

	end

	return temporalDifferenceErrorVector

end

function DeepDoubleStateActionRewardStateActionModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function DeepDoubleStateActionRewardStateActionModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function DeepDoubleStateActionRewardStateActionModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function DeepDoubleStateActionRewardStateActionModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepDoubleStateActionRewardStateActionModel