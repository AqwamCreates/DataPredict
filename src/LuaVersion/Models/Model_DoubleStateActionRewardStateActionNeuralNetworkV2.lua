local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

DoubleStateActionRewardStateActionNeuralNetworkModel = {}

DoubleStateActionRewardStateActionNeuralNetworkModel.__index = DoubleStateActionRewardStateActionNeuralNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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

function DoubleStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, averagingRate, discountFactor)

	local NewDoubleStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, discountFactor)

	setmetatable(NewDoubleStateActionRewardStateActionNeuralNetworkModel, DoubleStateActionRewardStateActionNeuralNetworkModel)

	NewDoubleStateActionRewardStateActionNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewDoubleStateActionRewardStateActionNeuralNetworkModel.ModelParameters == nil) then NewDoubleStateActionRewardStateActionNeuralNetworkModel:generateLayers() end

		local PrimaryModelParameters = NewDoubleStateActionRewardStateActionNeuralNetworkModel:getModelParameters()

		local targetVector = NewDoubleStateActionRewardStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		local dicountedVector = AqwamMatrixLibrary:multiply(NewDoubleStateActionRewardStateActionNeuralNetworkModel.discountFactor, targetVector)

		local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:train(previousFeatureVector, newTargetVector)

		local TargetModelParameters = NewDoubleStateActionRewardStateActionNeuralNetworkModel:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDoubleStateActionRewardStateActionNeuralNetworkModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:setModelParameters(TargetModelParameters, true)
		
		return newTargetVector

	end)

	return NewDoubleStateActionRewardStateActionNeuralNetworkModel

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, averagingRate, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.discountFactor =  discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

end

return DoubleStateActionRewardStateActionNeuralNetworkModel
