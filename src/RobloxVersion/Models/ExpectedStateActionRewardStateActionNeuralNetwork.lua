local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ExpectedStateActionRewardStateActionNeuralNetworkModel = {}

ExpectedStateActionRewardStateActionNeuralNetworkModel.__index = ExpectedStateActionRewardStateActionNeuralNetworkModel

setmetatable(ExpectedStateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultEpsilon = 0.5

function ExpectedStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, epsilon, discountFactor)

	local NewExpectedStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, discountFactor)

	setmetatable(NewExpectedStateActionRewardStateActionNeuralNetworkModel, ExpectedStateActionRewardStateActionNeuralNetworkModel)
	
	NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon = epsilon or defaultEpsilon

	NewExpectedStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local numberOfActions = #NewExpectedStateActionRewardStateActionNeuralNetworkModel.ClassesList

		local actionIndex = table.find(NewExpectedStateActionRewardStateActionNeuralNetworkModel.ClassesList, action)

		local predictedVector, maxQValue = NewExpectedStateActionRewardStateActionNeuralNetworkModel:predict(previousFeatureVector)

		local targetVector = NewExpectedStateActionRewardStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		for i = 1, numberOfActions, 1 do

			if (targetVector[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions += 1

		end

		local nonGreedyActionProbability = NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon / numberOfActions

		local greedyActionProbability = ((1 - NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for i, qValue in ipairs(targetVector[1]) do

			if (qValue == maxQValue) then

				expectedQValue += (qValue * greedyActionProbability)

			else

				expectedQValue += (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (NewExpectedStateActionRewardStateActionNeuralNetworkModel.discountFactor * expectedQValue)

		targetVector[1][actionIndex] = targetValue

		NewExpectedStateActionRewardStateActionNeuralNetworkModel:train(previousFeatureVector, targetVector)
		
		return targetValue

	end)

	return NewExpectedStateActionRewardStateActionNeuralNetworkModel

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, epsilon, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.epsilon = epsilon or self.epsilon

	self.discountFactor =  discountFactor or self.discountFactor

end

return ExpectedStateActionRewardStateActionNeuralNetworkModel
