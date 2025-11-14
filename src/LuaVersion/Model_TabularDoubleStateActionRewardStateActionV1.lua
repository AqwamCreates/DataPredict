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

function TabularDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleStateActionRewardStateActionModel, TabularDoubleStateActionRewardStateActionModel)
	
	NewTabularDoubleStateActionRewardStateActionModel:setName("TabularDoubleStateActionRewardStateActionV1")
	
	NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleStateActionRewardStateActionModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewTabularDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local learningRate = NewTabularDoubleStateActionRewardStateActionModel.learningRate
		
		local Optimizer = NewTabularDoubleStateActionRewardStateActionModel.Optimizer
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceError, stateIndex = NewTabularDoubleStateActionRewardStateActionModel:generateTemporalDifferenceError(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
		
		NewTabularDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		local ModelParameters = NewTabularDoubleStateActionRewardStateActionModel.ModelParameters
		
		local gradientValue = temporalDifferenceError

		if (Optimizer) then

			gradientValue = Optimizer:calculate(learningRate, {{gradientValue}})
			
			gradientValue = gradientValue[1][1]

		else
			
			gradientValue = AqwamTensorLibrary:multiply(learningRate, gradientValue)

		end
		
		local previousActionIndex = table.find(NewTabularDoubleStateActionRewardStateActionModel.ActionsList, previousAction)
		
		ModelParameters[stateIndex][previousActionIndex] = ModelParameters[stateIndex][previousActionIndex] + gradientValue
		
		NewTabularDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
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

function TabularDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self:getModelParameters()

end

function TabularDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)

	local ModelParametersArray = self.ModelParametersArray

	if (not ModelParametersArray[index]) then

		self:saveModelParametersFromModelParametersArray(index)

	end

	local CurrentModelParameters = ModelParametersArray[index]

	self:setModelParameters(CurrentModelParameters, true)

end

function TabularDoubleStateActionRewardStateActionModel:generateTemporalDifferenceError(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

	local discountFactor = self.discountFactor

	local EligibilityTrace = self.EligibilityTrace
	
	local StatesList = self:getStatesList()

	local ActionsList = self:getActionsList()

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
	
	local previousQVector = self:predict({{previousStateValue}}, true)

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)
	
	local currentQVector = self:predict({{currentStateValue}}, true)
	
	local StatesList = self:getStatesList()

	local ActionsList = self:getActionsList()

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

	return temporalDifferenceError, stateIndex

end

function TabularDoubleStateActionRewardStateActionModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function TabularDoubleStateActionRewardStateActionModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function TabularDoubleStateActionRewardStateActionModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function TabularDoubleStateActionRewardStateActionModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return TabularDoubleStateActionRewardStateActionModel
