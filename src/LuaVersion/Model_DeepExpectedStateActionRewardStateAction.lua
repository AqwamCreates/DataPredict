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

DeepExpectedStateActionRewardStateActionModel = {}

DeepExpectedStateActionRewardStateActionModel.__index = DeepExpectedStateActionRewardStateActionModel

setmetatable(DeepExpectedStateActionRewardStateActionModel, DeepReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

function DeepExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepExpectedStateActionRewardStateActionModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepExpectedStateActionRewardStateActionModel, DeepExpectedStateActionRewardStateActionModel)
	
	NewDeepExpectedStateActionRewardStateActionModel:setName("DeepExpectedStateActionRewardStateAction")
	
	NewDeepExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepExpectedStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewDeepExpectedStateActionRewardStateActionModel.epsilon
		
		local EligibilityTrace = NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local actionIndex = table.find(ClassesList, action)
		
		local previousVector = Model:forwardPropagate(previousFeatureVector)
		
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
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

		end
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep expected SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureVector, true)
		
		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewDeepExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepExpectedStateActionRewardStateActionModel

end

return DeepExpectedStateActionRewardStateActionModel
