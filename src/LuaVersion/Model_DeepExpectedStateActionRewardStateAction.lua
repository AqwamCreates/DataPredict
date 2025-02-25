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

DeepExpectedStateActionRewardStateActionModel = {}

DeepExpectedStateActionRewardStateActionModel.__index = DeepExpectedStateActionRewardStateActionModel

setmetatable(DeepExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

local defaultLambda = 0

function DeepExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepExpectedStateActionRewardStateActionModel, DeepExpectedStateActionRewardStateActionModel)
	
	NewDeepExpectedStateActionRewardStateActionModel:setName("DeepExpectedStateActionRewardStateAction")
	
	NewDeepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewDeepExpectedStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewDeepExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix

	NewDeepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepExpectedStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewDeepExpectedStateActionRewardStateActionModel.epsilon
		
		local lambda = NewDeepExpectedStateActionRewardStateActionModel.lambda

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local actionIndex = table.find(ClassesList, action)
		
		local previousVector = Model:forwardPropagate(previousFeatureVector)
		
		local targetVector = Model:forwardPropagate(currentFeatureVector)
		
		local maxQValue = targetVector[1][actionIndex]

		for i = 1, numberOfClasses, 1 do

			if (targetVector[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

		local nonGreedyActionProbability = epsilon / numberOfClasses

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(targetVector[1]) do

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

			local eligibilityTraceMatrix = NewDeepExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix

			if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

			eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)

			eligibilityTraceMatrix[1][actionIndex] = eligibilityTraceMatrix[1][actionIndex] + 1

			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTraceMatrix)

			NewDeepExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = eligibilityTraceMatrix

		end

		Model:forwardPropagate(previousFeatureVector, true, true)
		
		Model:backwardPropagate(temporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	NewDeepExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewDeepExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	return NewDeepExpectedStateActionRewardStateActionModel

end

return DeepExpectedStateActionRewardStateActionModel