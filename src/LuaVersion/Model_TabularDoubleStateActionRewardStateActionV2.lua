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

TabularDoubleStateActionRewardStateActionModel = {}

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
	
	NewTabularDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local averagingRate = NewTabularDoubleStateActionRewardStateActionModel.averagingRate
		
		local learningRate = NewTabularDoubleStateActionRewardStateActionModel.learningRate
		
		local discountFactor = NewTabularDoubleStateActionRewardStateActionModel.discountFactor
		
		local EligibilityTrace = NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local Optimizer = NewTabularDoubleStateActionRewardStateActionModel.Optimizer
		
		local ModelParameters = NewTabularDoubleStateActionRewardStateActionModel.ModelParameters
		
		local StatesList = NewTabularDoubleStateActionRewardStateActionModel:getStatesList()
		
		local previousQVector = NewTabularDoubleStateActionRewardStateActionModel:predict({{previousStateValue}}, true)

		local currentQVector = NewTabularDoubleStateActionRewardStateActionModel:predict({{currentStateValue}}, true)

		local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, currentQVector, (1 - terminalStateValue))

		local targetVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetVector, previousQVector)
		
		local averagingRateComplement = 1 - averagingRate
		
		if (EligibilityTrace) then
			
			local ActionsList = NewTabularDoubleStateActionRewardStateActionModel:getActionsList()

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
		
		local gradientTensor = temporalDifferenceErrorVector
		
		if (Optimizer) then
			
			gradientTensor = Optimizer:calculate(learningRate, gradientTensor)
			
		else
			
			gradientTensor = AqwamTensorLibrary:multiply(learningRate, gradientTensor)
			
		end
		
		local weightVector = {ModelParameters[stateIndex]}
		
		local targetVector = AqwamTensorLibrary:add(weightVector, gradientTensor)
		
		local multipliedPrimaryVector = AqwamTensorLibrary:multiply(averagingRate, weightVector)
		
		local multipliedTargetVector = AqwamTensorLibrary:multiply(averagingRateComplement, targetVector)
		
		ModelParameters[stateIndex] = AqwamTensorLibrary:add(multipliedPrimaryVector, multipliedTargetVector)[1]
		
		return temporalDifferenceErrorVector

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
