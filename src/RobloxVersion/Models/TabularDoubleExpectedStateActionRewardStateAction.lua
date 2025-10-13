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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local TabularReinforcementLearningBaseModel = require(script.Parent.TabularReinforcementLearningBaseModel)

TabularDoubleExpectedStateActionRewardStateActionModel = {}

TabularDoubleExpectedStateActionRewardStateActionModel.__index = TabularDoubleExpectedStateActionRewardStateActionModel

setmetatable(TabularDoubleExpectedStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local defaultEpsilon = 0.5

function TabularDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleExpectedStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTabularDoubleExpectedStateActionRewardStateActionModel, TabularDoubleExpectedStateActionRewardStateActionModel)
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel:setName("TabularDoubleExpectedStateActionRewardStateActionV2")
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewTabularDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local averagingRate = NewTabularDoubleExpectedStateActionRewardStateActionModel.averagingRate
		
		local discountFactor = NewTabularDoubleExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewTabularDoubleExpectedStateActionRewardStateActionModel.epsilon
		
		local EligibilityTrace = NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace
		
		local ModelParameters = NewTabularDoubleExpectedStateActionRewardStateActionModel.ModelParameters
		
		local StatesList = NewTabularDoubleExpectedStateActionRewardStateActionModel:getStatesList()

		local ActionsList = NewTabularDoubleExpectedStateActionRewardStateActionModel:getActionsList()
		
		local averagingRateComplement = 1 - averagingRate
		
		local numberOfActions = #ActionsList

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local actionIndex = table.find(ActionsList, action)
		
		local previousVector = NewTabularDoubleExpectedStateActionRewardStateActionModel:predict({{previousStateValue}}, true)
		
		local targetVector = NewTabularDoubleExpectedStateActionRewardStateActionModel:predict({{currentStateValue}}, true)
		
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

		local weightValue = ModelParameters[stateIndex][actionIndex]

		local newWeightValue = weightValue + (NewTabularDoubleExpectedStateActionRewardStateActionModel.learningRate * temporalDifferenceError)

		ModelParameters[stateIndex][actionIndex] = (averagingRate * weightValue) + (averagingRateComplement * newWeightValue)
		
		return temporalDifferenceError

	end)
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() 
		
		local EligibilityTrace = NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularDoubleExpectedStateActionRewardStateActionModel

end

return TabularDoubleExpectedStateActionRewardStateActionModel
