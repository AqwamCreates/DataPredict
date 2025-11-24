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

local TabularClippedDoubleQLearningModel = {}

TabularClippedDoubleQLearningModel.__index = TabularClippedDoubleQLearningModel

setmetatable(TabularClippedDoubleQLearningModel, TabularReinforcementLearningBaseModel)

function TabularClippedDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularClippedDoubleQLearningModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularClippedDoubleQLearningModel, TabularClippedDoubleQLearningModel)
	
	NewTabularClippedDoubleQLearningModel:setName("TabularClippedDoubleQLearning")
	
	NewTabularClippedDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewTabularClippedDoubleQLearningModel.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewTabularClippedDoubleQLearningModel:setCategoricalUpdateFunction(function(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)
		
		local learningRate = NewTabularClippedDoubleQLearningModel.learningRate

		local discountFactor = NewTabularClippedDoubleQLearningModel.discountFactor

		local EligibilityTrace = NewTabularClippedDoubleQLearningModel.EligibilityTrace
		
		local Optimizer = NewTabularClippedDoubleQLearningModel.Optimizer

		local ModelParametersArray = NewTabularClippedDoubleQLearningModel.ModelParametersArray
		
		local StatesList = NewTabularClippedDoubleQLearningModel:getStatesList()
		
		local ActionsList = NewTabularClippedDoubleQLearningModel:getActionsList()
		
		local previousStateValueVector = {{previousStateValue}}

		local maxQValueArray = {}

		for i = 1, 2, 1 do

			NewTabularClippedDoubleQLearningModel:setModelParameters(ModelParametersArray[i], true)

			local _, maxQValue = NewTabularClippedDoubleQLearningModel:predict(previousStateValueVector)

			table.insert(maxQValueArray, maxQValue[1][1])

			ModelParametersArray[i] = NewTabularClippedDoubleQLearningModel:getModelParameters(true)

		end

		local maxQValue = math.min(table.unpack(maxQValueArray))

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local actionIndex = table.find(ActionsList, previousAction)

		local temporalDifferenceErrorArray = {}
		
		local temporalDifferenceErrorMatrix

		if (EligibilityTrace) then 
			
			local numberOfStates = #StatesList

			local numberOfActions = #ActionsList
			
			local dimensionSizeArray = {numberOfStates, numberOfActions}
			
			temporalDifferenceErrorMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
			
			EligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray) 
			
		end

		for i = 1, 2, 1 do

			NewTabularClippedDoubleQLearningModel:setModelParameters(ModelParametersArray[i], true)

			local previousVector = NewTabularClippedDoubleQLearningModel:predict(previousStateValueVector, true)

			local lastValue = previousVector[1][actionIndex]

			local temporalDifferenceError = targetValue - lastValue

			if (EligibilityTrace) then 
				
				temporalDifferenceErrorMatrix[stateIndex][actionIndex] = temporalDifferenceError
				
				temporalDifferenceErrorMatrix = EligibilityTrace:calculate(temporalDifferenceErrorMatrix) 
				
				temporalDifferenceError = temporalDifferenceErrorMatrix[stateIndex][actionIndex]
				
			end
			
			local gradientValue = temporalDifferenceError
			
			if (Optimizer) then
				
				gradientValue = Optimizer:calculate(learningRate, {{gradientValue}})
				
				gradientValue = gradientValue[1][1]
				
			else
				
				gradientValue = learningRate * gradientValue
				
			end
			
			local ModelParameters = NewTabularClippedDoubleQLearningModel:getModelParameters(true)

			ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + gradientValue
			
			temporalDifferenceErrorArray[i] = temporalDifferenceError

		end
		
		local temporalDifferenceErrorVector = {temporalDifferenceErrorArray}

		return temporalDifferenceErrorVector

	end)
	
	NewTabularClippedDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local EligibilityTrace = NewTabularClippedDoubleQLearningModel.EligibilityTrace
		
		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewTabularClippedDoubleQLearningModel:setResetFunction(function()
		
		local EligibilityTrace = NewTabularClippedDoubleQLearningModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewTabularClippedDoubleQLearningModel

end

function TabularClippedDoubleQLearningModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function TabularClippedDoubleQLearningModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function TabularClippedDoubleQLearningModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function TabularClippedDoubleQLearningModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return TabularClippedDoubleQLearningModel
