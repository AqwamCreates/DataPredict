local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = {}

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.__index = DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

setmetatable(DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultEpsilon2 = 0.5

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	local NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)
	
	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon2 = epsilon2 or defaultEpsilon2
	
	setmetatable(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel)

	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ModelParametersArray = {}

	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local targetVector, targetValue = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:train(previousFeatureVector, targetVector)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

	end)

	return NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

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

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setModelParametersArray(ModelParameters1, ModelParameters2)

	if (ModelParameters1) or (ModelParameters2) then

		self.ModelParametersArray = {ModelParameters1, ModelParameters2}

	else

		self.ModelParametersArray = {}

	end

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParametersArray()

	return self.ModelParametersArray

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(index)

	local ModelParameters = self:getModelParameters()

	self.ModelParametersArray[index] = ModelParameters

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(index)

	local FirstModelParameters = self.ModelParametersArray[1]

	local SecondModelParameters = self.ModelParametersArray[2]

	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then

		self:generateLayers()

		self:saveModelParametersFromModelParametersArray(1)

		self:saveModelParametersFromModelParametersArray(2)

	end

	local CurrentModelParameters = self.ModelParametersArray[index]

	self:setModelParameters(CurrentModelParameters)

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local expectedQValue = 0

	local numberOfGreedyActions = 0

	local numberOfActions = #self.ClassesList

	local actionIndex = table.find(self.ClassesList, action)

	local predictedVector, maxQValue = self:predict(previousFeatureVector)

	local targetVector = self:predict(currentFeatureVector, true)

	for i = 1, numberOfActions, 1 do

		if (targetVector[1][i] ~= maxQValue) then continue end

		numberOfGreedyActions += 1

	end

	local nonGreedyActionProbability = self.epsilon2 / numberOfActions

	local greedyActionProbability = ((1 - self.epsilon2) / numberOfGreedyActions) + nonGreedyActionProbability

	for i, qValue in ipairs(targetVector[1]) do

		if (qValue == maxQValue) then

			expectedQValue += (qValue * greedyActionProbability)

		else

			expectedQValue += (qValue * nonGreedyActionProbability)

		end

	end

	local targetValue = rewardValue + (self.discountFactor * expectedQValue)

	targetVector[1][actionIndex] = targetValue

	return targetVector, targetValue

end

return DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel
