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
	
	NewTabularDoubleStateActionRewardStateActionModel.TargetModelParameters = parameterDictionary.TargetModelParameters
	
	NewTabularDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local Model = NewTabularDoubleStateActionRewardStateActionModel.Model

		local averagingRate = NewTabularDoubleStateActionRewardStateActionModel.averagingRate

		local discountFactor = NewTabularDoubleStateActionRewardStateActionModel.discountFactor

		local EligibilityTrace = NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace

		local TargetModelParameters = NewTabularDoubleStateActionRewardStateActionModel.TargetModelParameters

		local StatesList = NewTabularDoubleStateActionRewardStateActionModel:getStatesList()

		local ActionsList = NewTabularDoubleStateActionRewardStateActionModel:getActionsList()

		local PrimaryModelParameters = Model:getModelParameters(true)

		local averagingRateComplement = 1 - averagingRate
		
		if (not PrimaryModelParameters) then PrimaryModelParameters = Model:generateModelParameters() end

		if (not TargetModelParameters) then TargetModelParameters = NewTabularDoubleStateActionRewardStateActionModel:deepCopyTable(PrimaryModelParameters) end

		local primaryPreviousQVector = Model:getOutputMatrix(previousStateValue)

		local primaryCurrentActionIndex = table.find(ActionsList, currentAction)

		Model:setModelParameters(TargetModelParameters, true)

		local targetCurrentQVector = Model:getOutputMatrix(currentStateValue)

		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * targetCurrentQVector[1][primaryCurrentActionIndex])

		local stateIndex = table.find(StatesList, previousStateValue)

		local primaryPreviousActionIndex = table.find(ActionsList, previousAction)

		local primaryPreviousQValue = primaryPreviousQVector[1][primaryPreviousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue

		if (EligibilityTrace) then

			local numberOfStates = #StatesList

			local numberOfActions = #ActionsList

			local dimensionSizeArray = {numberOfStates, numberOfActions}

			local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

			temporalDifferenceErrorMatrix[stateIndex][primaryPreviousActionIndex] = temporalDifferenceError

			EligibilityTrace:increment(stateIndex, primaryPreviousActionIndex, discountFactor, dimensionSizeArray)

			temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)

			temporalDifferenceError = temporalDifferenceErrorMatrix[stateIndex][primaryPreviousActionIndex]

		end

		Model:setModelParameters(PrimaryModelParameters, true)

		Model:getOutputMatrix(previousStateValue, true)

		Model:update(-temporalDifferenceError, true)

		PrimaryModelParameters = Model:getModelParameters(true)

		local primaryQValue = PrimaryModelParameters[stateIndex][primaryPreviousActionIndex]

		local targetQValue = TargetModelParameters[stateIndex][primaryPreviousActionIndex]

		TargetModelParameters[stateIndex][primaryPreviousActionIndex] = (averagingRate * primaryQValue) + (averagingRateComplement * targetQValue)

		NewTabularDoubleStateActionRewardStateActionModel.TargetModelParameters = TargetModelParameters
		
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

function TabularDoubleStateActionRewardStateActionModel:setTargetModelParameters(TargetModelParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetModelParameters = TargetModelParameters

	else

		self.TargetModelParameters = self:deepCopyTable(TargetModelParameters)

	end

end

function TabularDoubleStateActionRewardStateActionModel:getTargetModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetModelParameters

	else

		return self:deepCopyTable(self.TargetModelParameters)

	end

end

return TabularDoubleStateActionRewardStateActionModel
