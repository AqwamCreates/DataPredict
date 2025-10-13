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

TabularStateActionRewardStateActionModel = {}

TabularStateActionRewardStateActionModel.__index = TabularStateActionRewardStateActionModel

setmetatable(TabularStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

function TabularStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularStateActionRewardStateActionModel, TabularStateActionRewardStateActionModel)
	
	NewTabularStateActionRewardStateActionModel:setName("TabularStateActionRewardStateAction")
	
	NewTabularStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local discountFactor = NewTabularStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewTabularStateActionRewardStateActionModel.EligibilityTrace
		
		local ModelParameters = NewTabularStateActionRewardStateActionModel.ModelParameters
		
		local StatesList = NewTabularStateActionRewardStateActionModel:getStatesList()
		
		local previousQVector = NewTabularStateActionRewardStateActionModel:predict({{previousStateValue}}, true)

		local currentQVector = NewTabularStateActionRewardStateActionModel:predict({{currentStateValue}}, true)

		local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, currentQVector, (1 - terminalStateValue))

		local targetVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetVector, previousQVector)
		
		if (EligibilityTrace) then
			
			local ActionsList = NewTabularStateActionRewardStateActionModel:getActionsList()

			local numberOfStates = #StatesList
			
			local numberOfActions = #ActionsList
			
			local actionIndex = table.find(ActionsList, action)

			local dimensionSizeArray = {numberOfStates, numberOfActions}

			local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

			temporalDifferenceErrorMatrix[stateIndex] = temporalDifferenceErrorVector[1]

			EligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray)

			temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)

			temporalDifferenceErrorVector = {temporalDifferenceErrorMatrix[stateIndex]}

		end
		
		local multipliedTemporalDifferenceErrorVector = AqwamTensorLibrary:multiply(NewTabularStateActionRewardStateActionModel.learningRate, temporalDifferenceErrorVector)
		
		ModelParameters[stateIndex] = AqwamTensorLibrary:add({ModelParameters[stateIndex]}, multipliedTemporalDifferenceErrorVector)[1]
		
		return temporalDifferenceErrorVector

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
