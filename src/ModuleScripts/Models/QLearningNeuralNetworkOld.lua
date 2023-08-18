local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

QLearningNeuralNetworkModel = {}

QLearningNeuralNetworkModel.__index = QLearningNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

setmetatable(QLearningNeuralNetworkModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

function QLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)

	local NewQLearningNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	setmetatable(NewQLearningNeuralNetworkModel, QLearningNeuralNetworkModel)

	NewQLearningNeuralNetworkModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewQLearningNeuralNetworkModel.epsilon = epsilon or defaultEpsilon

	NewQLearningNeuralNetworkModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewQLearningNeuralNetworkModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewQLearningNeuralNetworkModel.currentNumberOfEpisodes = 0

	NewQLearningNeuralNetworkModel.currentEpsilon = epsilon or defaultEpsilon

	NewQLearningNeuralNetworkModel.previousFeatureVector = nil

	return NewQLearningNeuralNetworkModel

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

function QLearningNeuralNetworkModel:calculateTargetModelParameters(currentFeatureVector, action, target)
	
	local actionIndex = table.find(self.ClassesList, action)
	
	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(action)

	local forwardPropagateTable, zTable = self:forwardPropagate(currentFeatureVector)

	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]
	
	--[[
	
	for column = 1, #allOutputsMatrix[1], 1 do
		
		if (column == actionIndex) then
			
			allOutputsMatrix[1][column] = target
			
		else
			
			allOutputsMatrix[1][column] = 0
			
		end
		
	end
	
	--]]
	
	logisticMatrix[1][actionIndex] = target

	local lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local backwardPropagateTable = self:backPropagate(lossMatrix, zTable)

	local deltaTable = self:calculateDelta(forwardPropagateTable, backwardPropagateTable)

	local TargetModelParameters = self:gradientDescent(self.learningRate, deltaTable, 1)
	
	return TargetModelParameters
	
end

function QLearningNeuralNetworkModel:update(previousFeatureVector, currentFeatureVector, action, rewardValue)
	
	if (self.ModelParameters == nil) then self:generateLayers() end

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local target = rewardValue + (self.discountFactor * maxQValue)

	local MainModelParameters = self:getModelParameters()

	local TargetModelParameters = self:calculateTargetModelParameters(previousFeatureVector, action, target)

	self:setModelParameters(TargetModelParameters) 

	local targetValue = self:predict(previousFeatureVector)

	targetValue = {{targetValue}}

	self:setModelParameters(MainModelParameters)

	self:train(previousFeatureVector, targetValue)

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
	
	return action, highestProbability

end

return QLearningNeuralNetworkModel
