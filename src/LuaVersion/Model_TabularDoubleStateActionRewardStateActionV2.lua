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

local TabularDoubleStateActionRewardStateActionModel = {}

TabularDoubleStateActionRewardStateActionModel.__index = TabularDoubleStateActionRewardStateActionModel

setmetatable(TabularDoubleStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

function TabularDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleStateActionRewardStateActionModel, TabularDoubleStateActionRewardStateActionModel)
	
	NewTabularDoubleStateActionRewardStateActionModel:setName("TabularDoubleStateActionRewardStateActionV2")
	
	NewTabularDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local averagingRate = NewTabularDoubleStateActionRewardStateActionModel.averagingRate
		
		local learningRate = NewTabularDoubleStateActionRewardStateActionModel.learningRate
		
		local discountFactor = NewTabularDoubleStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local Optimizer = NewTabularDoubleStateActionRewardStateActionModel.Optimizer
		
		local ModelParameters = NewTabularDoubleStateActionRewardStateActionModel.ModelParameters
		
		local StatesList = NewTabularDoubleStateActionRewardStateActionModel:getStatesList()

		local ActionsList = NewTabularDoubleStateActionRewardStateActionModel:getActionsList()
		
		local averagingRateComplement = 1 - averagingRate

		local previousQVector = NewTabularDoubleStateActionRewardStateActionModel:predict({{previousStateValue}}, true)

		local currentQVector = NewTabularDoubleStateActionRewardStateActionModel:predict({{currentStateValue}}, true)

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
		
		local weightValue = ModelParameters[stateIndex][previousActionIndex]

		local newWeightValue = weightValue + gradientValue

		ModelParameters[stateIndex][previousActionIndex] = (averagingRate * weightValue) + (averagingRateComplement * newWeightValue)
		
		return temporalDifferenceError

	end)
	
	NewTabularDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularDoubleStateActionRewardStateActionModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularDoubleStateActionRewardStateActionModel

end

return TabularDoubleStateActionRewardStateActionModel
