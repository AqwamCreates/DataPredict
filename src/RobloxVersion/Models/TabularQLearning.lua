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

TabularQLearningModel = {}

TabularQLearningModel.__index = TabularQLearningModel

setmetatable(TabularQLearningModel, TabularReinforcementLearningBaseModel)

local defaultLambda = 0

function TabularQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularQLearning = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularQLearning, TabularQLearningModel)
	
	NewTabularQLearning:setName("TabularQLearning")
	
	NewTabularQLearning.lambda = parameterDictionary.lambda or defaultLambda
	
	NewTabularQLearning.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix
	
	NewTabularQLearning:setCategoricalUpdateFunction(function(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)
		
		local discountFactor = NewTabularQLearning.discountFactor
		
		local lambda = NewTabularQLearning.lambda

		local _, maxQValue = NewTabularQLearning:predict(currentStateValue)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])
		
		local StatesList = NewTabularQLearning:getStatesList()
		
		local ActionsList = NewTabularQLearning:getActionsList()
		
		local ModelParameters = NewTabularQLearning.ModelParameters
		
		local stateIndex = table.find(StatesList, previousStateValue)

		local actionIndex = table.find(ActionsList, action)

		local lastValue = ModelParameters[stateIndex][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		if (lambda ~= 0) then
			
			local eligibilityTraceMatrix = NewTabularQLearning.eligibilityTraceMatrix
			
			if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor({#StatesList, #ActionsList}, 0) end
			
			eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)
			
			local eligibilityTraceValue = eligibilityTraceMatrix[stateIndex][actionIndex] + 1
			
			eligibilityTraceMatrix[stateIndex][actionIndex] = eligibilityTraceValue
			
			temporalDifferenceError = temporalDifferenceError * eligibilityTraceValue
			
			NewTabularQLearning.eligibilityTraceMatrix = eligibilityTraceMatrix
			
		end
		
		ModelParameters[stateIndex][actionIndex] = ModelParameters[stateIndex][actionIndex] + (NewTabularQLearning.learningRate * temporalDifferenceError)
		
		return temporalDifferenceError

	end)
	
	NewTabularQLearning:setEpisodeUpdateFunction(function(terminalStateValue)
		
		NewTabularQLearning.eligibilityTraceMatrix = nil
		
	end)

	NewTabularQLearning:setResetFunction(function()
		
		NewTabularQLearning.eligibilityTraceMatrix = nil
		
	end)

	return NewTabularQLearning

end

return TabularQLearningModel
