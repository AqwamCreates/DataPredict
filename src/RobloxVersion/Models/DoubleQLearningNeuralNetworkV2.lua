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

function DoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, averagingRate)

	local NewDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleQLearningNeuralNetworkModel, DoubleQLearningNeuralNetworkModel)
	
	NewDoubleQLearningNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewDoubleQLearningNeuralNetworkModel.ModelParameters == nil) then NewDoubleQLearningNeuralNetworkModel:generateLayers() end

		local PrimaryModelParameters = NewDoubleQLearningNeuralNetworkModel:getModelParameters()

		local predictedValue, maxQValue = NewDoubleQLearningNeuralNetworkModel:predict(currentFeatureVector)

		local targetValue = rewardValue + (NewDoubleQLearningNeuralNetworkModel.discountFactor * maxQValue[1][1])

		local targetVector = NewDoubleQLearningNeuralNetworkModel:predict(previousFeatureVector, true)

		local actionIndex = table.find(NewDoubleQLearningNeuralNetworkModel.ClassesList, action)

		targetVector[1][actionIndex] = targetValue

		NewDoubleQLearningNeuralNetworkModel:train(previousFeatureVector, targetVector)

		local TargetModelParameters = NewDoubleQLearningNeuralNetworkModel:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDoubleQLearningNeuralNetworkModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDoubleQLearningNeuralNetworkModel:setModelParameters(TargetModelParameters, true)
		
		return targetValue

	end)
	
	return NewDoubleQLearningNeuralNetworkModel

end

function DoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, averagingRate)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor
	
	self.averagingRate = averagingRate or self.averagingRate

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return DoubleQLearningNeuralNetworkModel
