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

local TabularDoubleQLearningModel = {}

TabularDoubleQLearningModel.__index = TabularDoubleQLearningModel

setmetatable(TabularDoubleQLearningModel, TabularReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

function TabularDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleQLearningModel, TabularDoubleQLearningModel)
	
	NewTabularDoubleQLearningModel:setName("TabularDoubleQLearningV2")
	
	NewTabularDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTabularDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local Model = NewTabularDoubleQLearningModel.Model
		
		local averagingRate = NewTabularDoubleQLearningModel.averagingRate
		
		local discountFactor = NewTabularDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewTabularDoubleQLearningModel.EligibilityTrace
		
		local StatesList = NewTabularDoubleQLearningModel:getStatesList()

		local ActionsList = NewTabularDoubleQLearningModel:getActionsList()
		
		local averagingRateComplement = 1 - averagingRate

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
		
		local OldModelParameters = Model:getModelParameters(true)
		
		local oldWeightValue = OldModelParameters[stateIndex][actionIndex]
		
		Model:update(-temporalDifferenceError, true)
		
		local NewModelParameters = Model:getModelParameters(true)
		
		local newWeightValue = NewModelParameters[stateIndex][actionIndex]
		
		NewModelParameters[stateIndex][actionIndex] = (averagingRate * oldWeightValue) + (averagingRateComplement * newWeightValue)
		
		Model:setModelParameters(NewModelParameters, true)
		
		return temporalDifferenceError

	end)
	
	NewTabularDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularDoubleQLearningModel.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularDoubleQLearningModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularDoubleQLearningModel

end

return TabularDoubleQLearningModel
