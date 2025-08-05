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

TabularStateActionRewardStateActionModel = {}

TabularStateActionRewardStateActionModel.__index = TabularStateActionRewardStateActionModel

setmetatable(TabularStateActionRewardStateActionModel, TabularReinforcementLearningBaseModel)

local defaultLambda = 0

function TabularStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTabularStateActionRewardStateActionModel = TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewTabularStateActionRewardStateActionModel, TabularStateActionRewardStateActionModel)
	
	NewTabularStateActionRewardStateActionModel:setName("TabularStateActionRewardStateActionModel")
	
	NewTabularStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewTabularStateActionRewardStateActionModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix
	
	NewTabularStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousState, action, rewardValue, currentState, terminalStateValue)
		
		local discountFactor = NewTabularStateActionRewardStateActionModel.discountFactor
		
		local lambda = NewTabularStateActionRewardStateActionModel.lambda
		
		local previousQVector = NewTabularStateActionRewardStateActionModel:predict(previousState, true)

		local currentQVector = NewTabularStateActionRewardStateActionModel:predict(currentState, true)

		local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, currentQVector, (1 - terminalStateValue))

		local targetVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)
		
		local StatesList = NewTabularStateActionRewardStateActionModel:getStatesList()
		
		local ActionsList = NewTabularStateActionRewardStateActionModel:getActionsList()
		
		local ModelParameters = NewTabularStateActionRewardStateActionModel.ModelParameters
		
		local stateIndex = table.find(StatesList, previousState)

		local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetVector, previousQVector)
		
		if (lambda ~= 0) then
			
			local actionIndex = table.find(ActionsList, action)
			
			local eligibilityTraceMatrix = NewTabularStateActionRewardStateActionModel.eligibilityTraceMatrix
			
			if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor({#StatesList, #ActionsList}, 0) end
			
			eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)
			
			local eligibilityTraceValue = eligibilityTraceMatrix[stateIndex][actionIndex] + 1
			
			eligibilityTraceMatrix[stateIndex][actionIndex] = eligibilityTraceValue

			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTraceMatrix)
			
			NewTabularStateActionRewardStateActionModel.eligibilityTraceMatrix = eligibilityTraceMatrix
			
		end
		
		local modifiedTemporalDifferenceErrorVector = AqwamTensorLibrary:multiply(NewTabularStateActionRewardStateActionModel.learningRate, temporalDifferenceErrorVector)
		
		ModelParameters[stateIndex] = AqwamTensorLibrary:add({ModelParameters[stateIndex]}, modifiedTemporalDifferenceErrorVector)[1]
		
		return temporalDifferenceErrorVector

	end)
	
	NewTabularStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		NewTabularStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	NewTabularStateActionRewardStateActionModel:setResetFunction(function()
		
		NewTabularStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	return NewTabularStateActionRewardStateActionModel

end

return TabularStateActionRewardStateActionModel
