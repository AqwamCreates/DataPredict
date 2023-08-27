local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ExpectedStateActionRewardStateActionNeuralNetworkModel = {}

ExpectedStateActionRewardStateActionNeuralNetworkModel.__index = ExpectedStateActionRewardStateActionNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

setmetatable(ExpectedStateActionRewardStateActionNeuralNetworkModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultEpsilon2 = 0.5

local defaultMaxNumberOfIterations = 1

local defaultExperienceReplayBatchSize = 32

local defaultMaxExperienceReplayBufferSize = 100

local defaultNumberOfReinforcementsForExperienceReplayUpdate = 1

function ExpectedStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewExpectedStateActionRewardStateActionNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	NewExpectedStateActionRewardStateActionNeuralNetworkModel:setPrintOutput(false)

	setmetatable(NewExpectedStateActionRewardStateActionNeuralNetworkModel, ExpectedStateActionRewardStateActionNeuralNetworkModel)

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon = epsilon or defaultEpsilon

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewExpectedStateActionRewardStateActionNeuralNetworkModel.epsilon2 = epsilon2 or defaultEpsilon2

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.currentNumberOfEpisodes = 0

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.currentEpsilon = epsilon or defaultEpsilon

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.previousFeatureVector = nil

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.printReinforcementOutput = true

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.replayBufferArray = {}

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.experienceReplayBatchSize = defaultExperienceReplayBatchSize

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.useExperienceReplay = false

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.maxExperienceReplayBufferSize = defaultMaxExperienceReplayBufferSize

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.numberOfReinforcementsForExperienceReplayUpdate = defaultNumberOfReinforcementsForExperienceReplayUpdate

	NewExpectedStateActionRewardStateActionNeuralNetworkModel.numberOfReinforcements = 0

	return NewExpectedStateActionRewardStateActionNeuralNetworkModel

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:setExperienceReplay(useExperienceReplay, experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)

	self.useExperienceReplay = self:getBooleanOrDefaultOption(useExperienceReplay, self.useExperienceReplay)

	self.experienceReplayBatchSize = experienceReplayBatchSize or self.experienceReplayBatchSize

	self.numberOfReinforcementsForExperienceReplayUpdate = numberOfReinforcementsForExperienceReplayUpdate or self.numberOfReinforcementsForExperienceReplayUpdate 

	self.maxExperienceReplayBufferSize = maxExperienceReplayBufferSize or self.maxExperienceReplayBufferSize

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, epsilon2, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor
	
	self.epsilon2 = epsilon2 or self.epsilon2

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	if (self.ModelParameters == nil) then self:generateLayers() end
	
	local expectedQValue = 0

	local numberOfGreedyActions = 0

	local numberOfActions = #self.ClassesList

	local actionIndex = table.find(self.ClassesList, action)

	local predictedVector, maxQValue = self:predict(previousFeatureVector)

	local targetVector = self:predict(currentFeatureVector, true)
	
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
	
	local newTargetValue = rewardValue + (self.discountFactor * expectedQValue)

	targetVector[1][actionIndex] = newTargetValue

	self:train(previousFeatureVector, targetVector)

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:sampleBatch()

	local batch = {}

	for i = 1, self.experienceReplayBatchSize, 1 do

		local index = Random.new():NextInteger(1, #self.replayBufferArray)

		table.insert(batch, self.replayBufferArray[index])

	end

	return batch

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:experienceReplayUpdate()

	if (#self.replayBufferArray < self.experienceReplayBatchSize) then return nil end

	local experienceReplayBatch = self:sampleBatch()

	for _, experience in ipairs(experienceReplayBatch) do -- (s1, a, r, s2)

		self:update(experience[1], experience[2], experience[3], experience[4])

	end

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:reset()

	self.numberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	self.replayBufferArray = {}

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

end

function ExpectedStateActionRewardStateActionNeuralNetworkModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

	if (self.ModelParameters == nil) then self:generateLayers() end

	self.currentNumberOfEpisodes = (self.currentNumberOfEpisodes + 1) % self.maxNumberOfEpisodes

	if (self.currentNumberOfEpisodes == 0) then

		self.currentEpsilon *= self.epsilonDecayFactor

	end

	local action

	local actionVector

	local highestProbability

	local highestProbabilityVector

	local allOutputsMatrix

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < self.currentEpsilon) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		allOutputsMatrix = self:predict(currentFeatureVector, true)

		actionVector, highestProbabilityVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

		action = actionVector[1][1]

		highestProbability = highestProbabilityVector[1][1]

	end

	if (self.previousFeatureVector) then self:update(self.previousFeatureVector, action, rewardValue, currentFeatureVector) end

	if (self.useExperienceReplay) and (self.previousFeatureVector) then 

		self.numberOfReinforcements = (self.numberOfReinforcements + 1) % self.numberOfReinforcementsForExperienceReplayUpdate

		if (self.numberOfReinforcements == 0) then self:experienceReplayUpdate() end

		local experience = {self.previousFeatureVector, action, rewardValue, currentFeatureVector}

		table.insert(self.replayBufferArray, experience)

		if (#self.replayBufferArray >= self.maxExperienceReplayBufferSize) then table.remove(self.replayBufferArray, 1) end

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput == true) then print("Current Number Of Episodes: " .. self.currentNumberOfEpisodes .. "\t\tCurrent Epsilon: " .. self.currentEpsilon) end

	if (returnOriginalOutput == true) then return allOutputsMatrix end

	return action, highestProbability

end

return ExpectedStateActionRewardStateActionNeuralNetworkModel
