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

local TabularQLearningModel = {}

TabularQLearningModel.__index = TabularQLearningModel

setmetatable(TabularQLearningModel, TabularReinforcementLearningBaseModel)

function TabularQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularQLearningModel, TabularQLearningModel)
	
	NewTabularQLearningModel:setName("TabularQLearning")
	
	NewTabularQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local Model = NewTabularQLearningModel.Model
		
		local discountFactor = NewTabularQLearningModel.discountFactor
		
		local EligibilityTrace = NewTabularQLearningModel.EligibilityTrace
		
		local StatesList = NewTabularQLearningModel:getStatesList()

		local ActionsList = NewTabularQLearningModel:getActionsList()

		local _, maxQValue = Model:predict(currentStateValue)
		
		local lastQVector = Model:getOutputMatrix(previousStateValue, true)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local actionIndex = table.find(ActionsList, previousAction)

		local lastValue = lastQVector[1][actionIndex]

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
		
		Model:update(-temporalDifferenceError, true)
		
		return temporalDifferenceError

	end)
	
	NewTabularQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularQLearningModel.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularQLearningModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularQLearningModel

end

return TabularQLearningModel
