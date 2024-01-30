local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = {}

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.__index = DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

setmetatable(DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultEpsilon2 = 0.5

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


function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor, averagingRate)

	local NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel)
	
	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon2 = epsilon2 or defaultEpsilon2
	
	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ModelParameters == nil) then NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:generateLayers() end

		local PrimaryModelParameters = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParameters()

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local numberOfActions = #NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ClassesList

		local actionIndex = table.find(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ClassesList, action)

		local predictedVector, maxQValue = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:predict(previousFeatureVector)

		local targetVector = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		for i = 1, numberOfActions, 1 do

			if (targetVector[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions += 1

		end

		local nonGreedyActionProbability = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon2 / numberOfActions

		local greedyActionProbability = ((1 - NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon2) / numberOfGreedyActions) + nonGreedyActionProbability

		for i, qValue in ipairs(targetVector[1]) do

			if (qValue == maxQValue) then

				expectedQValue += (qValue * greedyActionProbability)

			else

				expectedQValue += (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.discountFactor * expectedQValue)

		targetVector[1][actionIndex] = targetValue

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:train(previousFeatureVector, targetVector)

		local TargetModelParameters = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParameters()

		TargetModelParameters = rateAverageModelParameters(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setModelParameters(TargetModelParameters)
		
		return targetValue

	end)

	return NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor, averagingRate)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor
	
	self.epsilon2 = epsilon2 or self.epsilon2

	self.discountFactor =  discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel
