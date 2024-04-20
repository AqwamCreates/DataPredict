local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

QLearningNeuralNetworkModel = {}

QLearningNeuralNetworkModel.__index = QLearningNeuralNetworkModel

setmetatable(QLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function QLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, discountFactor)

	local NewQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, discountFactor)
	
	setmetatable(NewQLearningNeuralNetworkModel, QLearningNeuralNetworkModel)
	
	NewQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local predictedValue, maxQValue = NewQLearningNeuralNetworkModel:predict(currentFeatureVector)

		local targetValue = rewardValue + (NewQLearningNeuralNetworkModel.discountFactor * maxQValue[1][1])

		local numberOfClasses = #NewQLearningNeuralNetworkModel:getClassesList()

		local previousVector = NewQLearningNeuralNetworkModel:predict(previousFeatureVector, true)

		local actionIndex = table.find(NewQLearningNeuralNetworkModel.ClassesList, action)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

		lossVector[1][actionIndex] = temporalDifferenceError
		
		NewQLearningNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)

		NewQLearningNeuralNetworkModel:backPropagate(lossVector, true)
		
		return temporalDifferenceError

	end)

	return NewQLearningNeuralNetworkModel

end

function QLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, discountFactor)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.discountFactor =  discountFactor or self.discountFactor

end

return QLearningNeuralNetworkModel
