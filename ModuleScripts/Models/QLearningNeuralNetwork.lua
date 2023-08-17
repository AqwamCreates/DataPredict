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

local defaultExperienceReplayBatchSize = 32

local defaultMaxExperienceReplayBufferSize = 100

local defaultNumberOfReinforcementsForExperienceReplayUpdate = 0

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
	
	NewQLearningNeuralNetworkModel.replayBufferArray = {}
	
	NewQLearningNeuralNetworkModel.experienceReplayBatchSize = defaultExperienceReplayBatchSize
	
	NewQLearningNeuralNetworkModel.useExperienceReplay = false
	
	NewQLearningNeuralNetworkModel.maxExperienceReplayBufferSize = defaultMaxExperienceReplayBufferSize
	
	NewQLearningNeuralNetworkModel.numberOfReinforcementsForExperienceReplayUpdate = defaultNumberOfReinforcementsForExperienceReplayUpdate
	
	NewQLearningNeuralNetworkModel.numberOfReinforcements = 0

	return NewQLearningNeuralNetworkModel

end

function QLearningNeuralNetworkModel:setExperienceReplay(useExperienceReplay, experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)

	self.useExperienceReplay = self:getBooleanOrDefaultOption(useExperienceReplay, self.useExperienceReplay)
	
	self.experienceReplayBatchSize = experienceReplayBatchSize or self.experienceReplayBatchSize
	
	self.numberOfReinforcementsForExperienceReplayUpdate = numberOfReinforcementsForExperienceReplayUpdate or self.numberOfReinforcementsForExperienceReplayUpdate 
	
	self.maxExperienceReplayBufferSize = maxExperienceReplayBufferSize or self.maxExperienceReplayBufferSize
	
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

function QLearningNeuralNetworkModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	if (self.ModelParameters == nil) then self:generateLayers() end

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local target = rewardValue + (self.discountFactor * maxQValue[1][1])

	local targetVector = self:predict(previousFeatureVector, true)

	local actionIndex = table.find(self.ClassesList, action)

	targetVector[1][actionIndex] = target

	self:train(previousFeatureVector, targetVector)
	
end

function QLearningNeuralNetworkModel:sampleBatch()
	
	local batch = {}

	for i = 1, self.experienceReplayBatchSize, 1 do
		
		local index = Random.new():NextInteger(1, #self.replayBuffer)
		
		table.insert(batch, self.replayBuffer[index])
		
	end

	return batch
	
end

function QLearningNeuralNetworkModel:experienceReplayUpdate()
	
	if (#self.replayBufferArray < self.experienceReplayBatchSize) then return nil end
	
	local experienceReplayBatch = self:sampleBatch()

	for _, experience in ipairs(experienceReplayBatch) do -- (s1, a, r, s2)
		
		self:update(experience[1], experience[2], experience[3], experience[4])
		
	end
	
end

function QLearningNeuralNetworkModel:reset()
	
	self.numberOfReinforcements = 0
	
	self.currentNumberOfEpisodes = 0
	
	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	self.replayBufferArray = {}
	
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

	local actionVector
	
	local highestProbability
	
	local highestProbabilityVector

	local randomProbability = Random.new():NextNumber()
	
	if (randomProbability < self.epsilon) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]
		
		highestProbabilityVector = randomProbability

	else

		actionVector, highestProbabilityVector = self:predict(currentFeatureVector)
		
		action = actionVector[1][1]
		
		highestProbability = highestProbabilityVector[1][1]

	end

	self:update(self.previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	if (self.useExperienceReplay) then 
		
		self.numberOfReinforcements = (self.numberOfReinforcements + 1) % self.numberOfReinforcementsForExperienceReplayUpdate

		if (self.numberOfReinforcements == 0) then self:experienceReplayUpdate() end
		
		local experience = {self.previousFeatureVector, action, rewardValue, currentFeatureVector}

		table.insert(self.replayBufferArray, experience)
		
		if (#self.replayBufferArray >= self.maxExperienceReplayBufferSize) then table.remove(self.replayBufferArray, 1) end
		
	end
	
	if (self.printReinforcementOutput == true) then print("Current Number Of Episodes: " .. self.currentNumberOfEpisodes .. "\t\tCurrent Epsilon: " .. self.currentEpsilon) end
	
	return action, highestProbability

end

return QLearningNeuralNetworkModel
