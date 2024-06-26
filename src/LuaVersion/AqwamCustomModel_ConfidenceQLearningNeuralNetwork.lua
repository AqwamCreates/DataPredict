--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local LogisticRegression = require("Model_LogisticRegression")

local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

ConfidenceQLearningNeuralNetwork = {}

ConfidenceQLearningNeuralNetwork.__index = ConfidenceQLearningNeuralNetwork

setmetatable(ConfidenceQLearningNeuralNetwork, ReinforcementLearningNeuralNetworkBaseModel)

-- Do not multiply confidenceValue with target! Otherwise, it will cause poor performance!

function ConfidenceQLearningNeuralNetwork.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor, confidenceLearningRate)

	local NewConfidenceQLearningNeuralNetwork = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewConfidenceQLearningNeuralNetwork, ConfidenceQLearningNeuralNetwork)
	
	confidenceLearningRate = confidenceLearningRate
	
	local NewLogisticRegression = LogisticRegression.new(1, confidenceLearningRate, "Tanh")
	
	NewLogisticRegression:setPrintOutput(false)
	
	NewConfidenceQLearningNeuralNetwork:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		if (NewConfidenceQLearningNeuralNetwork.ModelParameters == nil) then NewConfidenceQLearningNeuralNetwork:generateLayers() end

		local predictedValueVector = NewConfidenceQLearningNeuralNetwork:predict(currentFeatureVector, true)
		
		local maxQValue = math.max(table.unpack(table.unpack(predictedValueVector)))
		
		if (NewLogisticRegression:getModelParameters() ~= nil) then
			
			local confidenceValue = NewLogisticRegression:predict(predictedValueVector)
			
			local target = (rewardValue * confidenceValue[1][1]) + (NewConfidenceQLearningNeuralNetwork.discountFactor * confidenceValue[1][1] * maxQValue)

			local targetVector = NewConfidenceQLearningNeuralNetwork:predict(previousFeatureVector, true)

			local actionIndex = table.find(NewConfidenceQLearningNeuralNetwork.ClassesList, action)

			targetVector[1][actionIndex] = target
			
			NewConfidenceQLearningNeuralNetwork:train(previousFeatureVector, targetVector)
			
		end
		
		NewLogisticRegression:train(predictedValueVector, {{rewardValue}})
		
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
