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

function TabularDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleStateActionRewardStateActionModel, TabularDoubleStateActionRewardStateActionModel)
	
	NewTabularDoubleStateActionRewardStateActionModel:setName("TabularDoubleStateActionRewardStateActionV1")
	
	NewTabularDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleStateActionRewardStateActionModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewTabularDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local learningRate = NewTabularDoubleStateActionRewardStateActionModel.learningRate
		
		local Optimizer = NewTabularDoubleStateActionRewardStateActionModel.Optimizer
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceErrorVector, stateIndex = NewTabularDoubleStateActionRewardStateActionModel:generateTemporalDifferenceError(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
		
		NewTabularDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		local ModelParameters = NewTabularDoubleStateActionRewardStateActionModel.ModelParameters

		if (Optimizer) then

			temporalDifferenceErrorVector = Optimizer:calculate(learningRate, temporalDifferenceErrorVector)

		else
			
			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(learningRate, temporalDifferenceErrorVector)

		end
		
		ModelParameters[stateIndex] = AqwamTensorLibrary:add({ModelParameters[stateIndex]}, temporalDifferenceErrorVector)[1]
		
		NewTabularDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
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

function TabularDoubleStateActionRewardStateActionModel:generateTemporalDifferenceError(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

	local discountFactor = self.discountFactor

	local EligibilityTrace = self.EligibilityTrace
	
	local StatesList = self:getStatesList()

	local ActionsList = self:getActionsList()

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
	
	local previousQVector = self:predict({{previousStateValue}}, true)

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)
	
	local currentQVector = self:predict({{currentStateValue}}, true)

	local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, currentQVector, (1 - terminalStateValue))

	local targetVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)

	local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetVector, previousQVector)
	
	local stateIndex = table.find(StatesList, previousStateValue)

	if (EligibilityTrace) then

		local numberOfStates = #StatesList

		local numberOfActions = #ActionsList

		local dimensionSizeArray = {numberOfStates, numberOfActions}
		
		local actionIndex = table.find(ActionsList, action)

		local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

		temporalDifferenceErrorMatrix[stateIndex] = temporalDifferenceErrorVector[1]

		EligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray)

		temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)

		temporalDifferenceErrorVector = {temporalDifferenceErrorMatrix[stateIndex]}

	end

	return temporalDifferenceErrorVector, stateIndex

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
