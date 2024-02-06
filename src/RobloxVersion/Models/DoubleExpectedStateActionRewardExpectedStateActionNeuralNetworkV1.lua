local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DoubleStateActionRewardStateActionNeuralNetworkModel = {}

DoubleStateActionRewardStateActionNeuralNetworkModel.__index = DoubleStateActionRewardStateActionNeuralNetworkModel

setmetatable(DoubleStateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function DoubleStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewDoubleStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleStateActionRewardStateActionNeuralNetworkModel, DoubleStateActionRewardStateActionNeuralNetworkModel)

	NewDoubleStateActionRewardStateActionNeuralNetworkModel.ModelParametersArray = {}

	NewDoubleStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local lossVector = NewDoubleStateActionRewardStateActionNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)
		
		NewDoubleStateActionRewardStateActionNeuralNetworkModel:backPropagate(lossVector, true)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return lossVector

	end)

	return NewDoubleStateActionRewardStateActionNeuralNetworkModel

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(index)

	local ModelParameters = self:getModelParameters()

	self.ModelParametersArray[index] = ModelParameters

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(index)

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

function DoubleStateActionRewardStateActionNeuralNetworkModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local targetVector = self:predict(currentFeatureVector, true)

	local dicountedVector = AqwamMatrixLibrary:multiply(self.discountFactor, targetVector)

	local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedVector)
	
	local previousVector = self:predict(previousFeatureVector, true)
	
	local lossVector = AqwamMatrixLibrary:subtract(newTargetVector, previousVector)

	return lossVector

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setModelParameters1(ModelParameters1)

	self.ModelParametersArray[1] = ModelParameters1

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return DoubleStateActionRewardStateActionNeuralNetworkModel
