local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

StateActionRewardStateActionNeuralNetworkModel = {}

StateActionRewardStateActionNeuralNetworkModel.__index = StateActionRewardStateActionNeuralNetworkModel

setmetatable(StateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function StateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewStateActionRewardStateActionNeuralNetworkModel, StateActionRewardStateActionNeuralNetworkModel)

	NewStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local targetVector = NewStateActionRewardStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		local dicountedVector = AqwamMatrixLibrary:multiply(NewStateActionRewardStateActionNeuralNetworkModel.discountFactor, targetVector)

		local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedVector)

		NewStateActionRewardStateActionNeuralNetworkModel:train(previousFeatureVector, newTargetVector)
		
		return newTargetVector

	end)

	return NewStateActionRewardStateActionNeuralNetworkModel

end

function StateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return StateActionRewardStateActionNeuralNetworkModel
