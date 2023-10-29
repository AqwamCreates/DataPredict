local Models = script.Parent.Parent.Models

local LogisticRegression = require(Models.LogisticRegression)

local ReinforcementLearningNeuralNetworkBaseModel = require(Models.ReinforcementLearningNeuralNetworkBaseModel)

ConfidenceQLearningNeuralNetwork = {}

ConfidenceQLearningNeuralNetwork.__index = ConfidenceQLearningNeuralNetwork

setmetatable(ConfidenceQLearningNeuralNetwork, ReinforcementLearningNeuralNetworkBaseModel)

function ConfidenceQLearningNeuralNetwork.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)

	local NewConfidenceQLearningNeuralNetwork = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewConfidenceQLearningNeuralNetwork, ConfidenceQLearningNeuralNetwork)
	
	local NewLogisticRegression = LogisticRegression.new(1)
	
	NewLogisticRegression:setPrintOutput(false)
	
	NewConfidenceQLearningNeuralNetwork:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewConfidenceQLearningNeuralNetwork.ModelParameters == nil) then NewConfidenceQLearningNeuralNetwork:generateLayers() end

		local predictedValueVector = NewConfidenceQLearningNeuralNetwork:predict(currentFeatureVector, true)
		
		local maxQValue = math.max(table.unpack(table.unpack(predictedValueVector)))
		
		if NewLogisticRegression:getModelParameters() ~= nil then
			
			local confidenceValue = NewLogisticRegression:predict(predictedValueVector)
			
			local target = (rewardValue * confidenceValue[1][1]) + (NewConfidenceQLearningNeuralNetwork.discountFactor * confidenceValue[1][1] * maxQValue)

			local targetVector = NewConfidenceQLearningNeuralNetwork:predict(previousFeatureVector, true)

			local actionIndex = table.find(NewConfidenceQLearningNeuralNetwork.ClassesList, action)

			targetVector[1][actionIndex] = target
			
			NewConfidenceQLearningNeuralNetwork:train(previousFeatureVector, targetVector)
			
		end
		
		NewLogisticRegression:train(predictedValueVector, {{math.min(rewardValue, 1)}})
		
	end)

	return NewConfidenceQLearningNeuralNetwork

end

function ConfidenceQLearningNeuralNetwork:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost
	
	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return ConfidenceQLearningNeuralNetwork
