local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepExpectedStateActionRewardStateActionModel = {}

DeepExpectedStateActionRewardStateActionModel.__index = DeepExpectedStateActionRewardStateActionModel

setmetatable(DeepExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

function DeepExpectedStateActionRewardStateActionModel.new(epsilon, discountFactor)

	local NewDeepExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepExpectedStateActionRewardStateActionModel, DeepExpectedStateActionRewardStateActionModel)
	
	NewDeepExpectedStateActionRewardStateActionModel.epsilon = epsilon or defaultEpsilon

	NewDeepExpectedStateActionRewardStateActionModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepExpectedStateActionRewardStateActionModel.Model

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local ClassesList = Model:getClassesList()

		local numberOfActions = #ClassesList

		local actionIndex = table.find(ClassesList, action)
		
		local previousVector = Model:predict(previousFeatureVector, true)
		
		local targetVector = Model:predict(currentFeatureVector, true)
		
		local maxQValue = math.max(table.unpack(targetVector[1]))

		for i = 1, numberOfActions, 1 do

			if (targetVector[1][i] ~= maxQValue) then continue end

			numberOfGreedyActions = numberOfGreedyActions + 1

		end

		local nonGreedyActionProbability = NewDeepExpectedStateActionRewardStateActionModel.epsilon / numberOfActions

		local greedyActionProbability = ((1 - NewDeepExpectedStateActionRewardStateActionModel.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for i, qValue in ipairs(targetVector[1]) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end
		
		local targetValue = rewardValue + (NewDeepExpectedStateActionRewardStateActionModel.discountFactor * expectedQValue)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, 0)
		
		lossVector[1][actionIndex] = temporalDifferenceError

		Model:forwardPropagate(previousFeatureVector, true)
		
		Model:backPropagate(lossVector, true)
		
		return temporalDifferenceError

	end)

	return NewDeepExpectedStateActionRewardStateActionModel

end

function DeepExpectedStateActionRewardStateActionModel:setParameters(epsilon, discountFactor)

	self.epsilon = epsilon or self.epsilon

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepExpectedStateActionRewardStateActionModel
