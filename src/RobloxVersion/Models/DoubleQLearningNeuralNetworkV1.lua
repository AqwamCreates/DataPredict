local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

DoubleQLearningNeuralNetworkModel = {}

DoubleQLearningNeuralNetworkModel.__index = DoubleQLearningNeuralNetworkModel

setmetatable(DoubleQLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function DoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleQLearningNeuralNetworkModel, DoubleQLearningNeuralNetworkModel)
	
	NewDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}
	
	NewDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDoubleQLearningNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local targetVector, targetValue = NewDoubleQLearningNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDoubleQLearningNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDoubleQLearningNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		NewDoubleQLearningNeuralNetworkModel:train(previousFeatureVector, targetVector)

		NewDoubleQLearningNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return targetValue
		
	end)

	return NewDoubleQLearningNeuralNetworkModel

end

function DoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function DoubleQLearningNeuralNetworkModel:setModelParametersArray(ModelParameters1, ModelParameters2)
	
	if (ModelParameters1) or (ModelParameters2) then
		
		self.ModelParametersArray = {ModelParameters1, ModelParameters2}
		
	else
		
		self.ModelParametersArray = {}
		
	end
	
end

function DoubleQLearningNeuralNetworkModel:getModelParametersArray()
	
	return self.ModelParametersArray
	
end

function DoubleQLearningNeuralNetworkModel:saveModelParametersFromModelParametersArray(index)

	local ModelParameters = self:getModelParameters()

	self.ModelParametersArray[index] = ModelParameters

end

function DoubleQLearningNeuralNetworkModel:loadModelParametersFromModelParametersArray(index)
	
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

function DoubleQLearningNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local targetValue = rewardValue + (self.discountFactor * maxQValue[1][1])

	local targetVector = self:predict(previousFeatureVector, true)

	local actionIndex = table.find(self.ClassesList, action)

	targetVector[1][actionIndex] = targetValue
	
	return targetVector, targetValue
	
end

return DoubleQLearningNeuralNetworkModel
