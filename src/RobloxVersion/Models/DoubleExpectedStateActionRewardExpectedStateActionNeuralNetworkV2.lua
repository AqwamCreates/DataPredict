local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = {}

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.__index = DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

setmetatable(DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultEpsilon = 0.5

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


function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, epsilon, averagingRate, discountFactor)

	local NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, discountFactor)

	setmetatable(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel)
	
	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon = epsilon or defaultEpsilon
	
	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ModelParameters == nil) then NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:generateLayers() end

		local PrimaryModelParameters = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParameters(true)

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local numberOfActions = #NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ClassesList

		local actionIndex = table.find(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ClassesList, action)

		local previousVector = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:predict(previousFeatureVector, true)

		local targetVector, maxQValue = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:predict(currentFeatureVector, false)

		for i = 1, numberOfActions, 1 do

			if (targetVector[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions += 1

		end

		local nonGreedyActionProbability = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon / numberOfActions

		local greedyActionProbability = ((1 - NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for i, qValue in ipairs(targetVector[1]) do

			if (qValue == maxQValue) then

				expectedQValue += (qValue * greedyActionProbability)

			else

				expectedQValue += (qValue * nonGreedyActionProbability)

			end

		end

		local numberOfClasses = #NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getClassesList()

		local targetValue = rewardValue + (NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.discountFactor * expectedQValue)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

		lossVector[1][actionIndex] = temporalDifferenceError

		local TargetModelParameters = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceError

	end)

	return NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, epsilon, averagingRate, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate
	
	self.epsilon = epsilon or self.epsilon

	self.discountFactor =  discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

end

return DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel
