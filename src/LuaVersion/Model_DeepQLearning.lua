--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

DeepQLearningModel = {}

DeepQLearningModel.__index = DeepQLearningModel

setmetatable(DeepQLearningModel, ReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepQLearningModel, DeepQLearningModel)
	
	NewDeepQLearningModel:setName("DeepQLearning")
	
	NewDeepQLearningModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewDeepQLearningModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix
	
	NewDeepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepQLearningModel.Model
		
		local discountFactor = NewDeepQLearningModel.discountFactor
		
		local lambda = NewDeepQLearningModel.lambda

		local _, maxQValue = Model:predict(currentFeatureVector)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local previousVector = Model:forwardPropagate(previousFeatureVector)

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorVector[1][actionIndex] = temporalDifferenceError
		
		if (lambda ~= 0) then
			
			local eligibilityTraceMatrix = NewDeepQLearningModel.eligibilityTraceMatrix
			
			if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end
			
			eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)
			
			eligibilityTraceMatrix[1][actionIndex] = eligibilityTraceMatrix[1][actionIndex] + 1
			
			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTraceMatrix)
			
			NewDeepQLearningModel.eligibilityTraceMatrix = eligibilityTraceMatrix
			
		end
		
		Model:forwardPropagate(previousFeatureVector, true, true)

		Model:backwardPropagate(temporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		NewDeepQLearningModel.eligibilityTraceMatrix = nil
		
	end)

	NewDeepQLearningModel:setResetFunction(function()
		
		NewDeepQLearningModel.eligibilityTraceMatrix = nil
		
	end)

	return NewDeepQLearningModel

end

return DeepQLearningModel