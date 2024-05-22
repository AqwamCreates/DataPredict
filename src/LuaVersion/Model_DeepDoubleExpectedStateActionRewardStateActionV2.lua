local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

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

function DeepDoubleExpectedStateActionRewardStateActionModel.new(maxNumberOfIterations, epsilon, averagingRate, discountFactor)

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(maxNumberOfIterations, discountFactor)

	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = epsilon or defaultEpsilon
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate = averagingRate or defaultAveragingRate

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model

		if (Model:getModelParameters() == nil) then Model:generateLayers() end

		local PrimaryModelParameters = NewDeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters(true)

		local expectedQValue = 0

		local numberOfGreedyActions = 0

		local numberOfActions = #NewDeepDoubleExpectedStateActionRewardStateActionModel.ClassesList

		local actionIndex = table.find(NewDeepDoubleExpectedStateActionRewardStateActionModel.ClassesList, action)

		local previousVector = NewDeepDoubleExpectedStateActionRewardStateActionModel:predict(previousFeatureVector, true)

		local targetVector = NewDeepDoubleExpectedStateActionRewardStateActionModel:predict(currentFeatureVector, true)
		
		local maxQValue = math.max(table.unpack(targetVector[1]))

		for i = 1, numberOfActions, 1 do

			if (targetVector[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

		local nonGreedyActionProbability = NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon / numberOfActions

		local greedyActionProbability = ((1 - NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for i, qValue in ipairs(targetVector[1]) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local numberOfClasses = #NewDeepDoubleExpectedStateActionRewardStateActionModel:getClassesList()

		local targetValue = rewardValue + (NewDeepDoubleExpectedStateActionRewardStateActionModel.discountFactor * expectedQValue)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

		lossVector[1][actionIndex] = temporalDifferenceError
		
		NewDeepDoubleExpectedStateActionRewardStateActionModel:forwardPropagate(previousFeatureVector, true)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:backPropagate(lossVector, true)

		local TargetModelParameters = NewDeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate, PrimaryModelParameters, TargetModelParameters)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceError

	end)

	return NewDeepDoubleExpectedStateActionRewardStateActionModel

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setParameters(epsilon, averagingRate, discountFactor)
	
	self.epsilon = epsilon or self.epsilon

	self.discountFactor =  discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

end

return DeepDoubleExpectedStateActionRewardStateActionModel
