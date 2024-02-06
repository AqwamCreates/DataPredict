local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = {}

DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.__index = DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

setmetatable(DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local defaultEpsilon2 = 0.5

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	local NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)
	
	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.epsilon2 = epsilon2 or defaultEpsilon2
	
	setmetatable(NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel, DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel)

	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel.ModelParametersArray = {}

	NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local lossVector, temporalDifferenceError = NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:backPropagate(lossVector, true)

		NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError

	end)

	return NewDoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor
	
	self.epsilon2 = epsilon2 or self.epsilon2

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

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

	self:setModelParameters(CurrentModelParameters, true)

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local expectedQValue = 0

	local numberOfGreedyActions = 0

	local numberOfActions = #self.ClassesList

	local actionIndex = table.find(self.ClassesList, action)

	local previousVector = self:predict(previousFeatureVector, true)

	local targetVector, maxQValue = self:predict(currentFeatureVector)

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
	
	local numberOfClasses = #self:getClassesList()

	local targetValue = rewardValue + (self.discountFactor * expectedQValue)
	
	local lastValue = previousVector[1][actionIndex]

	local temporalDifferenceError = targetValue - lastValue

	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

	lossVector[1][actionIndex] = temporalDifferenceError

	return lossVector, temporalDifferenceError

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setModelParameters1(ModelParameters1)
	
	self.ModelParametersArray[1] = ModelParameters1
	
end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return DoubleExpectedStateActionRewardExpectedStateActionNeuralNetworkModel
