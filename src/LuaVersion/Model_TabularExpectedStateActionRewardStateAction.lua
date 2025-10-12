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

local defaultEpsilon = 0.5

local defaultLambda = 0

function TabularExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularExpectedStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTabularExpectedStateActionRewardStateActionModel, TabularExpectedStateActionRewardStateActionModel)
	
	NewTabularExpectedStateActionRewardStateActionModel:setName("TabularExpectedStateActionRewardStateAction")
	
	NewTabularExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewTabularExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewTabularExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local discountFactor = NewTabularExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewTabularExpectedStateActionRewardStateActionModel.epsilon
		
		local EligibilityTrace = NewTabularExpectedStateActionRewardStateActionModel.EligibilityTrace
		
		local ModelParameters = NewTabularExpectedStateActionRewardStateActionModel.ModelParameters
		
		local StatesList = NewTabularExpectedStateActionRewardStateActionModel:getStatesList()

		local ActionsList = NewTabularExpectedStateActionRewardStateActionModel:getActionsList()
		
		local numberOfActions = #ActionsList

		local expectedQValue = 0

		local numberOfGreedyActions = 0

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
		
		if (EligibilityTrace) then
			
			local numberOfStates = #StatesList
			
			local dimensionSizeArray = {numberOfStates, numberOfActions}

			local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

			temporalDifferenceErrorMatrix[stateIndex][actionIndex] = temporalDifferenceError

			EligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray)

			temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)

			temporalDifferenceError = temporalDifferenceErrorMatrix[stateIndex][actionIndex]

		end

		ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + (NewTabularExpectedStateActionRewardStateActionModel.learningRate * temporalDifferenceError)
		
		return temporalDifferenceError

	end)
	
	NewTabularExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewTabularExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewTabularExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularExpectedStateActionRewardStateActionModel

end

return TabularExpectedStateActionRewardStateActionModel
