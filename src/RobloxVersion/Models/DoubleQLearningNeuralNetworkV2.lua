local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

DoubleQLearningNeuralNetworkModel = {}

DoubleQLearningNeuralNetworkModel.__index = DoubleQLearningNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

setmetatable(DoubleQLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultAveragingRate = 0.01

local function rateAverageModelParameters(averagingRate, PrimaryModelParameters, TargetModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local PrimaryModelParametersPart = AqwamMatrixLibrary:multiply(averagingRate, PrimaryModelParameters[layer])

		local TargetModelParametersPart = AqwamMatrixLibrary:multiply(averagingRateComplement, TargetModelParameters[layer])

		TargetModelParameters[layer] = AqwamMatrixLibrary:add(PrimaryModelParametersPart, TargetModelParametersPart)

	end

	return TargetModelParameters

end

function DoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, averagingRate, discountFactor)

	local NewDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, discountFactor)

	setmetatable(NewDoubleQLearningNeuralNetworkModel, DoubleQLearningNeuralNetworkModel)
	
	NewDoubleQLearningNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewDoubleQLearningNeuralNetworkModel.ModelParameters == nil) then NewDoubleQLearningNeuralNetworkModel:generateLayers() end

		local PrimaryModelParameters = NewDoubleQLearningNeuralNetworkModel:getModelParameters()

		local predictedValue, maxQValue = NewDoubleQLearningNeuralNetworkModel:predict(currentFeatureVector)

		local targetValue = rewardValue + (NewDoubleQLearningNeuralNetworkModel.discountFactor * maxQValue[1][1])

		local previousVector = NewDoubleQLearningNeuralNetworkModel:predict(previousFeatureVector, true)

		local actionIndex = table.find(NewDoubleQLearningNeuralNetworkModel.ClassesList, action)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		local numberOfClasses = #NewDoubleQLearningNeuralNetworkModel:getClassesList()

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

		lossVector[1][actionIndex] = temporalDifferenceError

		NewDoubleQLearningNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)

		NewDoubleQLearningNeuralNetworkModel:backPropagate(lossVector, true)

		local TargetModelParameters = NewDoubleQLearningNeuralNetworkModel:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDoubleQLearningNeuralNetworkModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDoubleQLearningNeuralNetworkModel:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceError

	end)
	
	return NewDoubleQLearningNeuralNetworkModel

end

function DoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, averagingRate, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.discountFactor =  discountFactor or self.discountFactor
	
	self.averagingRate = averagingRate or self.averagingRate

end

return DoubleQLearningNeuralNetworkModel
