local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

QLearningNeuralNetworkModel = {}

QLearningNeuralNetworkModel.__index = QLearningNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

setmetatable(QLearningNeuralNetworkModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

function QLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewQLearningNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)
	
	NewQLearningNeuralNetworkModel:setPrintOutput(false)

	setmetatable(NewQLearningNeuralNetworkModel, QLearningNeuralNetworkModel)

	NewQLearningNeuralNetworkModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewQLearningNeuralNetworkModel.epsilon = epsilon or defaultEpsilon

	NewQLearningNeuralNetworkModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewQLearningNeuralNetworkModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewQLearningNeuralNetworkModel.currentNumberOfEpisodes = 0

	NewQLearningNeuralNetworkModel.currentEpsilon = epsilon or defaultEpsilon

	NewQLearningNeuralNetworkModel.previousFeatureVector = nil
	
	NewQLearningNeuralNetworkModel.printReinforcementOutput = true

	return NewQLearningNeuralNetworkModel

end

function QLearningNeuralNetworkModel:setPrintReinforcementOutput(option)
	
	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)
	
end

function QLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost
	
	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function QLearningNeuralNetworkModel:update(previousFeatureVector, currentFeatureVector, action, rewardValue)

	if (self.ModelParameters == nil) then self:generateLayers() end

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local target = rewardValue + (self.discountFactor * maxQValue)
	
	local targetVector = self:predict(previousFeatureVector, true)
	
	local actionIndex = table.find(self.ClassesList, action)
	
	targetVector[1][actionIndex] = target
	
	self:train(previousFeatureVector, targetVector)

end

function QLearningNeuralNetworkModel:reset()
	
	self.currentNumberOfEpisodes = 0
	
	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

end

function QLearningNeuralNetworkModel:reinforce(currentFeatureVector, rewardValue)
	
	if (self.ModelParameters == nil) then self:generateLayers() end
	
	if (self.previousFeatureVector == nil) then

		self.previousFeatureVector = currentFeatureVector

		return nil

	end
	
	if (self.currentNumberOfEpisodes == 0) then

		self.currentEpsilon *= self.epsilonDecayFactor

	end

	self.currentNumberOfEpisodes = (self.currentNumberOfEpisodes + 1) % self.maxNumberOfEpisodes

	local action
	
	local highestProbability

	local randomProbability = Random.new():NextNumber()
	
	if (randomProbability < self.epsilon) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]
		
		highestProbability = randomProbability

	else

		action, highestProbability = self:predict(currentFeatureVector)

	end

	self:update(self.previousFeatureVector, currentFeatureVector, action, rewardValue)
	
	if (self.printReinforcementOutput == true) then print("Current Number Of Episodes: " .. self.currentNumberOfEpisodes .. "\t\tCurrent Epsilon: " .. self.currentEpsilon) end
	
	return action, highestProbability

end

return QLearningNeuralNetworkModel
