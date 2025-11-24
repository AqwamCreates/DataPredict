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

local TabularTemporalDifferenceModel = {}

TabularTemporalDifferenceModel.__index = TabularTemporalDifferenceModel

setmetatable(TabularTemporalDifferenceModel, TabularReinforcementLearningBaseModel)

function TabularTemporalDifferenceModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularTemporalDifferenceModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularTemporalDifferenceModel, TabularTemporalDifferenceModel)
	
	NewTabularTemporalDifferenceModel:setName("TabularTemporalDifference")
	
	NewTabularTemporalDifferenceModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularTemporalDifferenceModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local learningRate = NewTabularTemporalDifferenceModel.learningRate
		
		local discountFactor = NewTabularTemporalDifferenceModel.discountFactor
		
		local EligibilityTrace = NewTabularTemporalDifferenceModel.EligibilityTrace
		
		local Optimizer = NewTabularTemporalDifferenceModel.Optimizer
		
		local ModelParameters = NewTabularTemporalDifferenceModel.ModelParameters
		
		local StatesList = NewTabularTemporalDifferenceModel:getStatesList()
		
		local ActionsList = NewTabularTemporalDifferenceModel:getActionsList()
		
		local previousQVector = NewTabularTemporalDifferenceModel:predict({{previousStateValue}}, true)

		local currentQVector = NewTabularTemporalDifferenceModel:predict({{currentStateValue}}, true)

		local targetValue = rewardValue + (discountFactor * currentQVector[1][1] * (1 - terminalStateValue))

		local temporalDifferenceError = targetValue - previousQVector[1][1]
		
		local stateIndex = table.find(StatesList, previousStateValue)

		if (EligibilityTrace) then
			
			local actionIndex = table.find(ActionsList, previousAction)

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
		
		ModelParameters[stateIndex][1] = ModelParameters[stateIndex][1] + gradientValue
		
		return temporalDifferenceError

	end)
	
	NewTabularTemporalDifferenceModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularTemporalDifferenceModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularTemporalDifferenceModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularTemporalDifferenceModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularTemporalDifferenceModel

end

return TabularTemporalDifferenceModel
