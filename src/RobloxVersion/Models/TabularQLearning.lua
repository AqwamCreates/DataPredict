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

TabularQLearningModel = {}

TabularQLearningModel.__index = TabularQLearningModel

setmetatable(TabularQLearningModel, TabularReinforcementLearningBaseModel)

function TabularQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularQLearningModel, TabularQLearningModel)
	
	NewTabularQLearningModel:setName("TabularQLearning")
	
	NewTabularQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local discountFactor = NewTabularQLearningModel.discountFactor
		
		local EligibilityTrace = NewTabularQLearningModel.EligibilityTrace

		local ModelParameters = NewTabularQLearningModel.ModelParameters
		
		local StatesList = NewTabularQLearningModel:getStatesList()

		local ActionsList = NewTabularQLearningModel:getActionsList()

		local _, maxQValue = NewTabularQLearningModel:predict({{currentStateValue}})

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
		
		ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + (NewTabularQLearning.learningRate * temporalDifferenceError)
		
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
