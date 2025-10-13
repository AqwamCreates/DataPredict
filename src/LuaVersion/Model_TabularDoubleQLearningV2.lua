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

TabularDoubleQLearningModel = {}

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
	
	NewTabularDoubleQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local averagingRate = NewTabularDoubleQLearningModel.averagingRate
		
		local learningRate = NewTabularDoubleQLearningModel.learningRate
		
		local discountFactor = NewTabularDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewTabularDoubleQLearningModel.EligibilityTrace
		
		local Optimizer = NewTabularDoubleQLearningModel.Optimizer

		local ModelParameters = NewTabularDoubleQLearningModel.ModelParameters
		
		local StatesList = NewTabularDoubleQLearningModel:getStatesList()

		local ActionsList = NewTabularDoubleQLearningModel:getActionsList()
		
		local averagingRateComplement = 1 - averagingRate

		local _, maxQValue = NewTabularDoubleQLearningModel:predict({{currentStateValue}})

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
		
		local gradientValue = temporalDifferenceError

		if (Optimizer) then

			gradientValue = Optimizer:calculate(learningRate, {{gradientValue}})

			gradientValue = gradientValue[1][1]

		else

			gradientValue = learningRate * gradientValue

		end
		
		local weightValue = ModelParameters[stateIndex][actionIndex]
		
		local newWeightValue = weightValue + gradientValue
		
		ModelParameters[stateIndex][actionIndex] = (averagingRate * weightValue) + (averagingRateComplement * newWeightValue)
		
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
