local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

function DeepDoubleExpectedStateActionRewardStateActionModel.new(epsilon, discountFactor)

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = epsilon or defaultEpsilon
	
	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)

	NewDeepDoubleExpectedStateActionRewardStateActionModel.ModelParametersArray = {}

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDeepDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local lossVector, temporalDifferenceError = NewDeepDoubleExpectedStateActionRewardStateActionModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		Model:forwardPropagate(previousFeatureVector, true)

		Model:backPropagate(lossVector, true)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError

	end)

	return NewDeepDoubleExpectedStateActionRewardStateActionModel

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setParameters(epsilon, discountFactor)

	self.epsilon = epsilon or self.epsilon

	self.discountFactor =  discountFactor or self.discountFactor

end

function DeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)
	
	local Model = self.Model

	local FirstModelParameters = self.ModelParametersArray[1]

	local SecondModelParameters = self.ModelParametersArray[2]

	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then

		Model:generateLayers()

		self:saveModelParametersFromModelParametersArray(1)

		self:saveModelParametersFromModelParametersArray(2)

	end

	local CurrentModelParameters = self.ModelParametersArray[index]

	Model:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleExpectedStateActionRewardStateActionModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local Model = self.Model

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

	local nonGreedyActionProbability = self.epsilon / numberOfActions

	local greedyActionProbability = ((1 - self.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

	for i, qValue in ipairs(targetVector[1]) do

		if (qValue == maxQValue) then

			expectedQValue = expectedQValue + (qValue * greedyActionProbability)

		else

			expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

		end

	end

	local targetValue = rewardValue + (self.discountFactor * expectedQValue)
	
	local lastValue = previousVector[1][actionIndex]

	local temporalDifferenceError = targetValue - lastValue

	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, 0)

	lossVector[1][actionIndex] = temporalDifferenceError

	return lossVector, temporalDifferenceError

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters1(ModelParameters1)
	
	self.ModelParametersArray[1] = ModelParameters1
	
end

function DeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return DeepDoubleExpectedStateActionRewardStateActionModel
