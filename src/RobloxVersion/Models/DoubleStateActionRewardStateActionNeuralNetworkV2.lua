local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

DoubleStateActionRewardStateActionNeuralNetworkModel = {}

DoubleStateActionRewardStateActionNeuralNetworkModel.__index = DoubleStateActionRewardStateActionNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

setmetatable(DoubleStateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

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


function DoubleStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, averagingRate)

	local NewDoubleStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleStateActionRewardStateActionNeuralNetworkModel, DoubleStateActionRewardStateActionNeuralNetworkModel)

	NewDoubleStateActionRewardStateActionNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewDoubleStateActionRewardStateActionNeuralNetworkModel.ModelParameters == nil) then NewDoubleStateActionRewardStateActionNeuralNetworkModel:generateLayers() end

		local PrimaryModelParameters = NewDoubleStateActionRewardStateActionNeuralNetworkModel:getModelParameters()

		local targetVector = NewDoubleStateActionRewardStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		local dicountedVector = AqwamMatrixLibrary:multiply(NewDoubleStateActionRewardStateActionNeuralNetworkModel.discountFactor, targetVector)

		local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:train(previousFeatureVector, newTargetVector)

		local TargetModelParameters = NewDoubleStateActionRewardStateActionNeuralNetworkModel:getModelParameters()

		TargetModelParameters = rateAverageModelParameters(NewDoubleStateActionRewardStateActionNeuralNetworkModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:setModelParameters(TargetModelParameters)

	end)

	return NewDoubleStateActionRewardStateActionNeuralNetworkModel

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, averagingRate)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return DoubleStateActionRewardStateActionNeuralNetworkModel
