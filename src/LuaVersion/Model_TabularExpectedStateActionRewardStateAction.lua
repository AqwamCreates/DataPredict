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

local TabularReinforcementLearningBaseModel = require("Model_TabularReinforcementLearningBaseModel")

TabularExpectedStateActionRewardStateActionModel = {}

TabularExpectedStateActionRewardStateActionModel.__index = TabularExpectedStateActionRewardStateActionModel

setmetatable(TabularExpectedStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultLearningRate = 0.1

local defaultEpsilon = 0.5

local defaultLambda = 0

function TabularExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularExpectedStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTabularExpectedStateActionRewardStateActionModel, TabularExpectedStateActionRewardStateActionModel)
	
	NewTabularExpectedStateActionRewardStateActionModel:setName("TabularExpectedStateActionRewardStateAction")
	
	NewTabularExpectedStateActionRewardStateActionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewTabularExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewTabularExpectedStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewTabularExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix

	NewTabularExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local Model = NewTabularExpectedStateActionRewardStateActionModel.Model
		
		local discountFactor = NewTabularExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewTabularExpectedStateActionRewardStateActionModel.epsilon
		
		local lambda = NewTabularExpectedStateActionRewardStateActionModel.lambda
		
		local StatesList = NewTabularExpectedStateActionRewardStateActionModel:getStatesList()

		local ActionsList = NewTabularExpectedStateActionRewardStateActionModel:getActionsList()
		
		local ModelParameters = NewTabularExpectedStateActionRewardStateActionModel.ModelParameters

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local numberOfActions = #ActionsList

		local actionIndex = table.find(ActionsList, action)
		
		local previousVector = NewTabularExpectedStateActionRewardStateActionModel:predict({{previousStateValue}}, true)
		
		local targetVector = NewTabularExpectedStateActionRewardStateActionModel:predict({{currentStateValue}}, true)
		
		local maxQValue = AqwamTensorLibrary:findMaximumValue(targetVector)
		
		local stateIndex = table.find(StatesList, previousStateValue)
		
		local actionIndex = table.find(ActionsList, action)

		local unwrappedTargetVector = targetVector[1]

		for i = 1, numberOfActions, 1 do

			if (unwrappedTargetVector[i] == maxQValue) then

				numberOfGreedyActions = numberOfGreedyActions + 1

			end

		end

		local nonGreedyActionProbability = epsilon / numberOfActions

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
		
		local outputDimensionSizeArray = {1, numberOfActions}
		
		if (lambda ~= 0) then

			local eligibilityTraceMatrix = NewTabularExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix

			if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor({#StatesList, #ActionsList}, 0) end

			eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)

			local eligibilityTraceValue = eligibilityTraceMatrix[stateIndex][actionIndex] + 1

			eligibilityTraceMatrix[stateIndex][actionIndex] = eligibilityTraceValue

			temporalDifferenceError = temporalDifferenceError * eligibilityTraceValue

			NewTabularExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = eligibilityTraceMatrix

		end

		ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + (NewTabularExpectedStateActionRewardStateActionModel.learningRate * temporalDifferenceError)
		
		return temporalDifferenceError

	end)
	
	NewTabularExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewTabularExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	NewTabularExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewTabularExpectedStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	return NewTabularExpectedStateActionRewardStateActionModel

end

return TabularExpectedStateActionRewardStateActionModel
