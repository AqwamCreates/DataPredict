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

TabularDoubleQLearningModel = {}

TabularDoubleQLearningModel.__index = TabularDoubleQLearningModel

setmetatable(TabularDoubleQLearningModel, TabularReinforcementLearningBaseModel)

function TabularDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleQLearningModel, TabularDoubleQLearningModel)
	
	NewTabularDoubleQLearningModel:setName("TabularDoubleQLearningV2")
	
	NewTabularDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleQLearningModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewTabularDoubleQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceError, stateIndex, actionIndex = NewTabularDoubleQLearningModel:generateTemporalDifferenceError(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
		
		NewTabularDoubleQLearningModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		local ModelParameters = NewTabularDoubleQLearningModel.ModelParameters
		
		ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + (NewTabularDoubleQLearningModel.learningRate * temporalDifferenceError)
		
		NewTabularDoubleQLearningModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
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

function TabularDoubleQLearningModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self:getModelParameters()

end

function TabularDoubleQLearningModel:loadModelParametersFromModelParametersArray(index)

	local ModelParametersArray = self.ModelParametersArray

	if (not ModelParametersArray[index]) then

		self:saveModelParametersFromModelParametersArray(index)

	end

	local CurrentModelParameters = ModelParametersArray[index]

	self:setModelParameters(CurrentModelParameters, true)

end

function TabularDoubleQLearningModel:generateTemporalDifferenceError(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

	local discountFactor = self.discountFactor

	local EligibilityTrace = self.EligibilityTrace
	
	local StatesList = self:getStatesList()

	local ActionsList = self:getActionsList()

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
	
	local previousVector = self:predict({{previousStateValue}}, true)

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)
	
	local _, maxQValue = self:predict({{currentStateValue}})
	
	local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])

	local stateIndex = table.find(StatesList, previousStateValue)

	local actionIndex = table.find(ActionsList, action)

	local lastValue = previousVector[1][actionIndex]

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

	return temporalDifferenceError, stateIndex, actionIndex

end

function TabularDoubleQLearningModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function TabularDoubleQLearningModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function TabularDoubleQLearningModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function TabularDoubleQLearningModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return TabularDoubleQLearningModel
