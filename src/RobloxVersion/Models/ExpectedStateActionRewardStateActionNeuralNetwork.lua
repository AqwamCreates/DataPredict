local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ExpectedStateActionRewardStateActionNeuralNetworkModel = {}

ExpectedStateActionRewardStateActionNeuralNetworkModel.__index = ExpectedStateActionRewardStateActionNeuralNetworkModel

setmetatable(ExpectedStateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultEpsilon2 = 0.5

function ExpectedStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	local NewExpectedStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewExpectedStateActionRewardStateActionNeuralNetworkModel, ExpectedStateActionRewardStateActionNeuralNetworkModel)
	
	NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon2 = epsilon2 or defaultEpsilon2

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

		local nonGreedyActionProbability = NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon2 / numberOfActions

		local greedyActionProbability = ((1 - NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon2) / numberOfGreedyActions) + nonGreedyActionProbability

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

function ExpectedStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor
	
	self.epsilon2 = epsilon2 or self.epsilon2

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return ExpectedStateActionRewardStateActionNeuralNetworkModel
