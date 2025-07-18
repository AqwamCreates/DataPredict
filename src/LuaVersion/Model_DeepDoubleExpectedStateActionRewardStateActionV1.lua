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

DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultLambda = 0

function DeepDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel:setName("DeepExpectedStateActionRewardStateActionV1")

	NewDeepDoubleExpectedStateActionRewardStateActionModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewDeepDoubleExpectedStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda

	NewDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorVector, temporalDifferenceError = NewDeepDoubleExpectedStateActionRewardStateActionModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError

	end)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewDeepDoubleExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	return NewDeepDoubleExpectedStateActionRewardStateActionModel

end

function DeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)

	local Model = self.Model

	local ModelParametersArray = self.ModelParametersArray

	if (not ModelParametersArray[index]) then

		Model:generateLayers()

		self:saveModelParametersFromModelParametersArray(index)

	end

	local CurrentModelParameters = ModelParametersArray[index]

	Model:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleExpectedStateActionRewardStateActionModel:generateTemporalDifferenceErrorVector(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
	
	local Model = self.Model
	
	local discountFactor = self.discountFactor
	
	local epsilon = self.epsilon
	
	local lambda = self.lambda

	local expectedQValue = 0

	local numberOfGreedyActions = 0
	
	local ClassesList = Model:getClassesList()

	local numberOfClasses = #ClassesList

	local actionIndex = table.find(ClassesList, action)
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

	local previousVector = Model:forwardPropagate(previousFeatureVector)
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

	local targetVector = Model:forwardPropagate(currentFeatureVector)

	local maxQValue = AqwamTensorLibrary:findMaximumValue(targetVector)

	local unwrappedTargetVector = targetVector[1]

	for i = 1, numberOfClasses, 1 do

		if (unwrappedTargetVector[i] == maxQValue) then

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

	end

	local nonGreedyActionProbability = epsilon / numberOfClasses

	local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

	for _, qValue in ipairs(unwrappedTargetVector) do

		if (qValue == maxQValue) then

			expectedQValue = expectedQValue + (qValue * greedyActionProbability)

		else

			expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

		end

	end

	local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)
	
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

function DeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepDoubleExpectedStateActionRewardStateActionModel