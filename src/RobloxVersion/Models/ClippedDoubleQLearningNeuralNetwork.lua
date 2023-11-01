local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

ClippedDoubleQLearningNeuralNetworkModel = {}

ClippedDoubleQLearningNeuralNetworkModel.__index = ClippedDoubleQLearningNeuralNetworkModel

setmetatable(ClippedDoubleQLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function ClippedDoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewClippedDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}
	
	NewClippedDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local maxQValues = {}

		for i = 1, 2, 1 do

			NewClippedDoubleQLearningNeuralNetworkModel:setModelParameters(NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray[i])

			local predictedValue, maxQValue = NewClippedDoubleQLearningNeuralNetworkModel:predict(currentFeatureVector)

			table.insert(maxQValues, maxQValue[1][1])

		end

		local maxQValue = math.min(table.unpack(maxQValues))

		local target = rewardValue + (NewClippedDoubleQLearningNeuralNetworkModel.discountFactor * maxQValue)

		local actionIndex = table.find(NewClippedDoubleQLearningNeuralNetworkModel.ClassesList, action)

		for i = 1, 2, 1 do

			NewClippedDoubleQLearningNeuralNetworkModel:setModelParameters(NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray[i])

			local targetVector = NewClippedDoubleQLearningNeuralNetworkModel:predict(previousFeatureVector, true)

			targetVector[1][actionIndex] = maxQValue

			NewClippedDoubleQLearningNeuralNetworkModel:train(previousFeatureVector, targetVector)

		end

	end)

	return NewClippedDoubleQLearningNeuralNetworkModel

end

function ClippedDoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function ClippedDoubleQLearningNeuralNetworkModel:setModelParametersArray(ModelParameters1, ModelParameters2)

	if (ModelParameters1) or (ModelParameters2) then

		self.ModelParametersArray = {ModelParameters1, ModelParameters2}

	else

		self.ModelParametersArray = {}

	end

end

function ClippedDoubleQLearningNeuralNetworkModel:getModelParametersArray()

	return self.ModelParametersArray

end

return ClippedDoubleQLearningNeuralNetworkModel
