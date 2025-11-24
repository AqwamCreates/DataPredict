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

local TabularStateActionRewardStateActionModel = {}

TabularStateActionRewardStateActionModel.__index = TabularStateActionRewardStateActionModel

setmetatable(TabularStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

function TabularStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularStateActionRewardStateActionModel, TabularStateActionRewardStateActionModel)
	
	NewTabularStateActionRewardStateActionModel:setName("TabularStateActionRewardStateAction")
	
	NewTabularStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local learningRate = NewTabularStateActionRewardStateActionModel.learningRate
		
		local discountFactor = NewTabularStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewTabularStateActionRewardStateActionModel.EligibilityTrace
		
		local Optimizer = NewTabularStateActionRewardStateActionModel.Optimizer
		
		local ModelParameters = NewTabularStateActionRewardStateActionModel.ModelParameters
		
		local StatesList = NewTabularStateActionRewardStateActionModel:getStatesList()
		
		local ActionsList = NewTabularStateActionRewardStateActionModel:getActionsList()
		
		local previousQVector = NewTabularStateActionRewardStateActionModel:predict({{previousStateValue}}, true)

		local currentQVector = NewTabularStateActionRewardStateActionModel:predict({{currentStateValue}}, true)

		local previousActionIndex = table.find(ActionsList, previousAction)

		local currentActionIndex = table.find(ActionsList, currentAction)
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local targetValue = rewardValue + (discountFactor * currentQVector[1][currentActionIndex] * (1 - terminalStateValue))

		local temporalDifferenceError = targetValue - previousQVector[1][previousActionIndex]
		
		if (EligibilityTrace) then

			local numberOfStates = #StatesList

			local numberOfActions = #ActionsList

			local dimensionSizeArray = {numberOfStates, numberOfActions}

			local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

			temporalDifferenceErrorMatrix[stateIndex][previousActionIndex] = temporalDifferenceError

			EligibilityTrace:increment(stateIndex, previousActionIndex, discountFactor, dimensionSizeArray)

			temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)

			temporalDifferenceError = temporalDifferenceErrorMatrix[stateIndex][previousActionIndex]

		end
		
		local gradientValue = temporalDifferenceError

		if (Optimizer) then

			gradientValue = Optimizer:calculate(learningRate, {{gradientValue}})

			gradientValue = gradientValue[1][1]

		else

			gradientValue = learningRate * gradientValue

		end

		ModelParameters[stateIndex][previousActionIndex] = ModelParameters[stateIndex][previousActionIndex] + gradientValue
		
		return temporalDifferenceError

	end)
	
	NewTabularStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularStateActionRewardStateActionModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularStateActionRewardStateActionModel

end

return TabularStateActionRewardStateActionModel
