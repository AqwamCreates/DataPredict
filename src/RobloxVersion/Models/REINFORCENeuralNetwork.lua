local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

REINFORCENeuralNetworkModel = {}

REINFORCENeuralNetworkModel.__index = REINFORCENeuralNetworkModel

setmetatable(REINFORCENeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function REINFORCENeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewREINFORCENeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewREINFORCENeuralNetworkModel, REINFORCENeuralNetworkModel)
	
	local policyGradientMatrix = {}
	
	NewREINFORCENeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local predictedVector = NewREINFORCENeuralNetworkModel:predict(previousFeatureVector, true)
		
		local logPredictedVector = AqwamMatrixLibrary:applyFunction(math.log, predictedVector)
		
		local targetVector = AqwamMatrixLibrary:multiply(logPredictedVector, rewardValue)

		table.insert(policyGradientMatrix, targetVector[1])

	end)
	
	NewREINFORCENeuralNetworkModel:setEpisodeUpdateFunction(function()
		
		local targetVector = AqwamMatrixLibrary:verticalMean(policyGradientMatrix)
		
		local numberOfNeurons = NewREINFORCENeuralNetworkModel.numberOfNeuronsTable[1] + NewREINFORCENeuralNetworkModel.hasBiasNeuronTable[1]
		
		local inputVector = {table.create(numberOfNeurons, 1)}
		
		NewREINFORCENeuralNetworkModel:train(inputVector, targetVector)
		
		table.clear(policyGradientMatrix)
		
	end)
	
	NewREINFORCENeuralNetworkModel:extendResetFunction(function()

		table.clear(policyGradientMatrix)
		
	end)

	return NewREINFORCENeuralNetworkModel

end

function REINFORCENeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return REINFORCENeuralNetworkModel
