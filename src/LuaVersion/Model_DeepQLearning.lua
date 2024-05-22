local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningBaseModel = require("ReinforcementLearningBaseModel")

DeepQLearningModel = {}

DeepQLearningModel.__index = DeepQLearningModel

setmetatable(DeepQLearningModel, ReinforcementLearningBaseModel)

function DeepQLearningModel.new(discountFactor)

	local NewDeepQLearningModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewDeepQLearningModel, DeepQLearningModel)
	
	NewDeepQLearningModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
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

		Model:backPropagate(lossVector, true)
		
		return temporalDifferenceError

	end)

	return NewDeepQLearningModel

end

function DeepQLearningModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepQLearningModel
