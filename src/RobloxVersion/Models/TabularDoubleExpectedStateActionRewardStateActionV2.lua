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

local TabularDoubleExpectedStateActionRewardStateActionModel = {}

TabularDoubleExpectedStateActionRewardStateActionModel.__index = TabularDoubleExpectedStateActionRewardStateActionModel

setmetatable(TabularDoubleExpectedStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local defaultEpsilon = 0.5

function TabularDoubleExpectedStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleExpectedStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTabularDoubleExpectedStateActionRewardStateActionModel, TabularDoubleExpectedStateActionRewardStateActionModel)
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel:setName("TabularDoubleExpectedStateActionRewardStateActionV2")
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleExpectedStateActionRewardStateActionModel.TargetModelParameters = parameterDictionary.TargetModelParameters

	NewTabularDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local Model = NewTabularDoubleExpectedStateActionRewardStateActionModel.Model
		
		local averagingRate = NewTabularDoubleExpectedStateActionRewardStateActionModel.averagingRate
		
		local discountFactor = NewTabularDoubleExpectedStateActionRewardStateActionModel.discountFactor
		
		local epsilon = NewTabularDoubleExpectedStateActionRewardStateActionModel.epsilon
		
		local EligibilityTrace = NewTabularDoubleExpectedStateActionRewardStateActionModel.EligibilityTrace
		
		local TargetModelParameters = NewTabularDoubleExpectedStateActionRewardStateActionModel.TargetModelParameters
		
		local StatesList = NewTabularDoubleExpectedStateActionRewardStateActionModel:getStatesList()

		local ActionsList = NewTabularDoubleExpectedStateActionRewardStateActionModel:getActionsList()
		
		local PrimaryModelParameters = Model:getModelParameters(true)
		
		local averagingRateComplement = 1 - averagingRate
		
		if (not PrimaryModelParameters) then PrimaryModelParameters = Model:generateModelParameters() end
		
		if (not TargetModelParameters) then TargetModelParameters = NewTabularDoubleExpectedStateActionRewardStateActionModel:deepCopyTable(PrimaryModelParameters) end
		
		local primaryPreviousQVector = Model:getOutputMatrix(previousStateValue)
		
		local primaryCurrentQVector = Model:getOutputMatrix(currentStateValue)

		local primaryCurrentActionIndex = table.find(ActionsList, currentAction)
		
		local unwrappedPrimaryCurrentVector = primaryCurrentQVector[1]
		
		local maximumPrimaryCurrentQValue = unwrappedPrimaryCurrentVector[primaryCurrentActionIndex]

		Model:setModelParameters(TargetModelParameters, true)

		local targetCurrentQVector = Model:getOutputMatrix(currentStateValue)
		
		local numberOfActions = #ActionsList

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local stateIndex = table.find(StatesList, previousStateValue)
		
		local primaryPreviousActionIndex = table.find(ActionsList, previousAction)

		for i = 1, numberOfActions do
			
			if (unwrappedPrimaryCurrentVector[i] == maximumPrimaryCurrentQValue) then
				
				numberOfGreedyActions = numberOfGreedyActions + 1
				
			end
			
		end

		local nonGreedyActionProbability = epsilon / numberOfActions

		local greedyActionProbability = ((1 - epsilon) / numberOfGreedyActions) + nonGreedyActionProbability
		
		local unwrappedTargetCurrentQVector = targetCurrentQVector[1]

		local actionProbability
		
		local isGreedy

		for i, targetCurrentQValue in ipairs(unwrappedTargetCurrentQVector) do
			
			isGreedy = (unwrappedPrimaryCurrentVector[i] == maximumPrimaryCurrentQValue)

			actionProbability = (isGreedy and greedyActionProbability) or nonGreedyActionProbability

			expectedQValue = expectedQValue + (targetCurrentQValue * actionProbability)

		end
		
		local targetQValue = rewardValue + (discountFactor * (1 - terminalStateValue) * expectedQValue)

		local primaryPreviousQValue = primaryPreviousQVector[1][primaryPreviousActionIndex]

		local temporalDifferenceError = targetQValue - primaryPreviousQValue
		
		if (EligibilityTrace) then
			
			local numberOfStates = #StatesList
			
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

		NewTabularDoubleExpectedStateActionRewardStateActionModel.TargetModelParameters = TargetModelParameters
		
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

function TabularDoubleExpectedStateActionRewardStateActionModel:setTargetModelParameters(TargetModelParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetModelParameters = TargetModelParameters

	else

		self.TargetModelParameters = self:deepCopyTable(TargetModelParameters)

	end

end

function TabularDoubleExpectedStateActionRewardStateActionModel:getTargetModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetModelParameters

	else

		return self:deepCopyTable(self.TargetModelParameters)

	end

end

return TabularDoubleExpectedStateActionRewardStateActionModel
