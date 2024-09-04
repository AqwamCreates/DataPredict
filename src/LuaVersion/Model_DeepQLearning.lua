--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

DeepQLearningModel = {}

DeepQLearningModel.__index = DeepQLearningModel

setmetatable(DeepQLearningModel, ReinforcementLearningBaseModel)

function DeepQLearningModel.new(discountFactor)

	local NewDeepQLearningModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewDeepQLearningModel, DeepQLearningModel)
	
	NewDeepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepQLearningModel.Model

		local predictedValue, maxQValue = Model:predict(currentFeatureVector)

		local targetValue = rewardValue + (NewDeepQLearningModel.discountFactor * maxQValue[1][1])
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local previousVector = Model:predict(previousFeatureVector, true)

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

		lossVector[1][actionIndex] = temporalDifferenceError
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:backwardPropagate(lossVector, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepQLearningModel:setEpisodeUpdateFunction(function() end)

	NewDeepQLearningModel:setResetFunction(function() end)

	return NewDeepQLearningModel

end

function DeepQLearningModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepQLearningModel