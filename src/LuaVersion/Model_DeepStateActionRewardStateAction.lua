local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

DeepStateActionRewardStateActionModel = {}

DeepStateActionRewardStateActionModel.__index = DeepStateActionRewardStateActionModel

setmetatable(DeepStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

function DeepStateActionRewardStateActionModel.new(discountFactor)

	local NewDeepStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepStateActionRewardStateActionModel, DeepStateActionRewardStateActionModel)

	NewDeepStateActionRewardStateActionModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepStateActionRewardStateActionModel.Model

		local qVector = Model:predict(currentFeatureVector, true)

		local discountedQVector = AqwamMatrixLibrary:multiply(NewDeepStateActionRewardStateActionModel.discountFactor, qVector)

		local targetVector = AqwamMatrixLibrary:add(rewardValue, discountedQVector)

		local previousQVector = Model:predict(previousFeatureVector, true)

		local temporalDifferenceVector = AqwamMatrixLibrary:subtract(targetVector, previousQVector)
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:backPropagate(temporalDifferenceVector, true)
		
		return temporalDifferenceVector

	end)

	return NewDeepStateActionRewardStateActionModel

end

function DeepStateActionRewardStateActionModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepStateActionRewardStateActionModel
