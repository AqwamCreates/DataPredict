local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

StateActionRewardStateActionNeuralNetworkModel = {}

StateActionRewardStateActionNeuralNetworkModel.__index = StateActionRewardStateActionNeuralNetworkModel

setmetatable(StateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function StateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, discountFactor)

	local NewStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, discountFactor)

	setmetatable(NewStateActionRewardStateActionNeuralNetworkModel, StateActionRewardStateActionNeuralNetworkModel)

	NewStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local qVector = NewStateActionRewardStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		local discountedQVector = AqwamMatrixLibrary:multiply(NewStateActionRewardStateActionNeuralNetworkModel.discountFactor, qVector)

		local targetVector = AqwamMatrixLibrary:add(rewardValue, discountedQVector)

		local previousQVector = NewStateActionRewardStateActionNeuralNetworkModel:predict(previousFeatureVector, true)

		local temporalDifferenceVector = AqwamMatrixLibrary:subtract(targetVector, previousQVector)
		
		NewStateActionRewardStateActionNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)

		NewStateActionRewardStateActionNeuralNetworkModel:backPropagate(temporalDifferenceVector, true)
		
		return temporalDifferenceVector

	end)

	return NewStateActionRewardStateActionNeuralNetworkModel

end

function StateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.discountFactor =  discountFactor or self.discountFactor

end

return StateActionRewardStateActionNeuralNetworkModel
