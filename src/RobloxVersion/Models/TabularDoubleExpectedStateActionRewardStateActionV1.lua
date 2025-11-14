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

TabularDoubleExpectedStateActionRewardStateActionModel = {}

TabularDoubleExpectedStateActionRewardStateActionModel.__index = TabularDoubleExpectedStateActionRewardStateActionModel

setmetatable(TabularDoubleExpectedStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

function TabularDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleExpectedStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleExpectedStateActionRewardStateActionModel, TabularDoubleExpectedStateActionRewardStateActionModel)
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel:setName("TabularDoubleExpectedStateActionRewardStateActionV1")
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local learningRate = NewTabularDoubleExpectedStateActionRewardStateActionModel.learningRate
		
		local Optimizer = NewTabularDoubleExpectedStateActionRewardStateActionModel.Optimizer
		
		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local temporalDifferenceError, stateIndex, actionIndex = NewTabularDoubleExpectedStateActionRewardStateActionModel:generateTemporalDifferenceError(previousStateValue, previousAction, rewardValue, currentStateValue, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
		
		NewTabularDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		local ModelParameters = NewTabularDoubleExpectedStateActionRewardStateActionModel.ModelParameters

		local gradientValue = temporalDifferenceError

		if (Optimizer) then

			gradientValue = Optimizer:calculate(learningRate, {{gradientValue}})

			gradientValue = gradientValue[1][1]

		else

			gradientValue = learningRate * gradientValue

		end

		ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + gradientValue
		
		NewTabularDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError

	end)
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularDoubleExpectedStateActionRewardStateActionModel

end

function TabularDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self:getModelParameters()

end

function TabularDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)

	local ModelParametersArray = self.ModelParametersArray

	if (not ModelParametersArray[index]) then

		self:saveModelParametersFromModelParametersArray(index)

	end

	local CurrentModelParameters = ModelParametersArray[index]

	self:setModelParameters(CurrentModelParameters, true)

end

function TabularDoubleExpectedStateActionRewardStateActionModel:generateTemporalDifferenceError(previousStateValue, previousAction, rewardValue, currentStateValue, terminalStateValue, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

	local discountFactor = self.discountFactor
	
	local epsilon = self.epsilon

	local EligibilityTrace = self.EligibilityTrace
	
	local StatesList = self:getStatesList()

	local ActionsList = self:getActionsList()

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
	
	local previousVector = self:predict({{previousStateValue}}, true)

	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)
	
	local targetVector = self:predict({{currentStateValue}}, true)

	local numberOfActions = #ActionsList

	local expectedQValue = 0

	local numberOfGreedyActions = 0

	local maxQValue = AqwamTensorLibrary:findMaximumValue(targetVector)

	local stateIndex = table.find(StatesList, previousStateValue)

	local actionIndex = table.find(ActionsList, previousAction)

	local unwrappedTargetVector = targetVector[1]

	for i = 1, numberOfActions, 1 do

		if (unwrappedTargetVector[i] == maxQValue) then

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

	end

	local nonGreedyActionProbability = epsilon / numberOfActions

	local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

	for _, qValue in ipairs(unwrappedTargetVector) do

		if (qValue == maxQValue) then

			expectedQValue = expectedQValue + (qValue * greedyActionProbability)

		else

			expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

		end

	end

	local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

	local lastValue = previousVector[1][actionIndex]

	local temporalDifferenceError = targetValue - lastValue

	if (EligibilityTrace) then

		local numberOfStates = #StatesList

		local dimensionSizeArray = {numberOfStates, numberOfActions}

		local temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

		temporalDifferenceErrorMatrix[stateIndex][actionIndex] = temporalDifferenceError

		EligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray)

		temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix)

		temporalDifferenceError = temporalDifferenceErrorMatrix[stateIndex][actionIndex]

	end

	return temporalDifferenceError, stateIndex, actionIndex

end

function TabularDoubleExpectedStateActionRewardStateActionModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function TabularDoubleExpectedStateActionRewardStateActionModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function TabularDoubleExpectedStateActionRewardStateActionModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function TabularDoubleExpectedStateActionRewardStateActionModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return TabularDoubleExpectedStateActionRewardStateActionModel
