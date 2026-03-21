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

local TabularDoubleQLearningModel = {}

TabularDoubleQLearningModel.__index = TabularDoubleQLearningModel

setmetatable(TabularDoubleQLearningModel, TabularReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

function TabularDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularDoubleQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularDoubleQLearningModel, TabularDoubleQLearningModel)
	
	NewTabularDoubleQLearningModel:setName("TabularDoubleQLearningV2")
	
	NewTabularDoubleQLearningModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewTabularDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularDoubleQLearningModel.TargetModelParameters = parameterDictionary.TargetModelParameters
	
	NewTabularDoubleQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local Model = NewTabularDoubleQLearningModel.Model
		
		local averagingRate = NewTabularDoubleQLearningModel.averagingRate
		
		local discountFactor = NewTabularDoubleQLearningModel.discountFactor
		
		local EligibilityTrace = NewTabularDoubleQLearningModel.EligibilityTrace
		
		local TargetModelParameters = NewTabularDoubleQLearningModel.TargetModelParameters
		
		local StatesList = NewTabularDoubleQLearningModel:getStatesList()

		local ActionsList = NewTabularDoubleQLearningModel:getActionsList()
		
		local PrimaryModelParameters = Model:getModelParameters(true)
		
		local averagingRateComplement = 1 - averagingRate

		if (not PrimaryModelParameters) then 

			Model:generateLayers()

			PrimaryModelParameters = Model:getModelParameters(true)

		end
		
		if (not TargetModelParameters) then TargetModelParameters = NewTabularDoubleQLearningModel:deepCopyTable(PrimaryModelParameters) end
		
		local primaryPreviousQVector = Model:getOutputMatrix(previousStateValue)
		
		local maximumPrimaryCurrentActionVector = Model:predict(currentStateValue)
		
		local primaryCurrentActionIndex = table.find(ActionsList, maximumPrimaryCurrentActionVector[1][1])
		
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
		
		NewTabularDoubleQLearningModel.TargetModelParameters = TargetModelParameters
		
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

function TabularDoubleQLearningModel:setTargetModelParameters(TargetModelParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.TargetModelParameters = TargetModelParameters

	else

		self.TargetModelParameters = self:deepCopyTable(TargetModelParameters)

	end

end

function TabularDoubleQLearningModel:getTargetModelParameters(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.TargetModelParameters

	else

		return self:deepCopyTable(self.TargetModelParameters)

	end

end

return TabularDoubleQLearningModel
