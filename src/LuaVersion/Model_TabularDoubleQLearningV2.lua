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

TabularDoubleQLearningV2Model = {}

TabularDoubleQLearningV2Model.__index = TabularDoubleQLearningV2Model

setmetatable(TabularDoubleQLearningV2Model, TabularReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

function TabularDoubleQLearningV2Model.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleQLearningV2 = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleQLearningV2, TabularDoubleQLearningV2Model)
	
	NewTabularDoubleQLearningV2:setName("TabularDoubleQLearningV2")
	
	NewTabularDoubleQLearningV2.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTabularDoubleQLearningV2.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleQLearningV2:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local averagingRate = NewTabularDoubleQLearningV2.averagingRate
		
		local discountFactor = NewTabularDoubleQLearningV2.discountFactor
		
		local EligibilityTrace = NewTabularDoubleQLearningV2.EligibilityTrace

		local ModelParameters = NewTabularDoubleQLearningV2.ModelParameters
		
		local StatesList = NewTabularDoubleQLearningV2:getStatesList()

		local ActionsList = NewTabularDoubleQLearningV2:getActionsList()
		
		local averagingRateComplement = 1 - averagingRate

		local _, maxQValue = NewTabularDoubleQLearningV2:predict({{currentStateValue}})

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local actionIndex = table.find(ActionsList, action)

		local lastValue = ModelParameters[stateIndex][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		if (EligibilityTrace) then
			
			local numberOfStates = #StatesList

			local numberOfActions = #ActionsList
			
			local dimensionSizeArray = {numberOfStates, numberOfActions}
			
			local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
			
			temporalDifferenceErrorMatrix[stateIndex][actionIndex] = temporalDifferenceError

			EligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray)

			temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)
			
			temporalDifferenceError = temporalDifferenceErrorMatrix[stateIndex][actionIndex]

		end
		
		local weightValue = ModelParameters[stateIndex][actionIndex]
		
		local newWeightValue = weightValue + (NewTabularDoubleQLearningV2.learningRate * temporalDifferenceError)
		
		ModelParameters[stateIndex][actionIndex] = (averagingRate * weightValue) + (averagingRateComplement * newWeightValue)
		
		return temporalDifferenceError

	end)
	
	NewTabularDoubleQLearningV2:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularDoubleQLearningV2.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularDoubleQLearningV2:setResetFunction(function()
		
		local EligibilityTrace = NewTabularDoubleQLearningV2.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularDoubleQLearningV2

end

return TabularDoubleQLearningV2Model
