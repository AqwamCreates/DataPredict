local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ClippedDoubleQLearningNeuralNetworkModel = {}

ClippedDoubleQLearningNeuralNetworkModel.__index = ClippedDoubleQLearningNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

setmetatable(ClippedDoubleQLearningNeuralNetworkModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

local defaultExperienceReplayBatchSize = 32

local defaultMaxExperienceReplayBufferSize = 100

local defaultNumberOfReinforcementsForExperienceReplayUpdate = 1

function ClippedDoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)

	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewClippedDoubleQLearningNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	NewClippedDoubleQLearningNeuralNetworkModel:setPrintOutput(false)

	setmetatable(NewClippedDoubleQLearningNeuralNetworkModel, ClippedDoubleQLearningNeuralNetworkModel)

	NewClippedDoubleQLearningNeuralNetworkModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewClippedDoubleQLearningNeuralNetworkModel.epsilon = epsilon or defaultEpsilon

	NewClippedDoubleQLearningNeuralNetworkModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewClippedDoubleQLearningNeuralNetworkModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewClippedDoubleQLearningNeuralNetworkModel.currentNumberOfEpisodes = 0

	NewClippedDoubleQLearningNeuralNetworkModel.currentEpsilon = epsilon or defaultEpsilon

	NewClippedDoubleQLearningNeuralNetworkModel.previousFeatureVector = nil

	NewClippedDoubleQLearningNeuralNetworkModel.printReinforcementOutput = true

	NewClippedDoubleQLearningNeuralNetworkModel.replayBufferArray = {}

	NewClippedDoubleQLearningNeuralNetworkModel.experienceReplayBatchSize = defaultExperienceReplayBatchSize

	NewClippedDoubleQLearningNeuralNetworkModel.useExperienceReplay = false

	NewClippedDoubleQLearningNeuralNetworkModel.maxExperienceReplayBufferSize = defaultMaxExperienceReplayBufferSize

	NewClippedDoubleQLearningNeuralNetworkModel.numberOfReinforcementsForExperienceReplayUpdate = defaultNumberOfReinforcementsForExperienceReplayUpdate

	NewClippedDoubleQLearningNeuralNetworkModel.numberOfReinforcements = 0

	NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}

	return NewClippedDoubleQLearningNeuralNetworkModel

end

function ClippedDoubleQLearningNeuralNetworkModel:setExperienceReplay(useExperienceReplay, experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)

	self.useExperienceReplay = self:getBooleanOrDefaultOption(useExperienceReplay, self.useExperienceReplay)

	self.experienceReplayBatchSize = experienceReplayBatchSize or self.experienceReplayBatchSize

	self.numberOfReinforcementsForExperienceReplayUpdate = numberOfReinforcementsForExperienceReplayUpdate or self.numberOfReinforcementsForExperienceReplayUpdate 

	self.maxExperienceReplayBufferSize = maxExperienceReplayBufferSize or self.maxExperienceReplayBufferSize

end

function ClippedDoubleQLearningNeuralNetworkModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ClippedDoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

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


function ClippedDoubleQLearningNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	self:setModelParameters(self.ModelParametersArray[1])

	local predictedValue1, maxQValue1 = self:predict(currentFeatureVector)

	self:setModelParameters(self.ModelParametersArray[2])

	local predictedValue2, maxQValue2 = self:predict(currentFeatureVector)

	maxQValue1 = maxQValue1[1][1]

	maxQValue2 = maxQValue2[1][1]

	local maxQValue = math.max(maxQValue1, maxQValue2)

	local target = rewardValue + (self.discountFactor * maxQValue)

	local targetVector = self:predict(previousFeatureVector, true)

	local actionIndex = table.find(self.ClassesList, action)

	targetVector[1][actionIndex] = target

	return targetVector

end

function ClippedDoubleQLearningNeuralNetworkModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local targetVector = self:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	self:train(previousFeatureVector, targetVector)

end

function ClippedDoubleQLearningNeuralNetworkModel:sampleBatch()

	local batch = {}

	for i = 1, self.experienceReplayBatchSize, 1 do

		local index = Random.new():NextInteger(1, #self.replayBufferArray)

		table.insert(batch, self.replayBufferArray[index])

	end

	return batch

end

function ClippedDoubleQLearningNeuralNetworkModel:experienceReplayUpdate()

	if (#self.replayBufferArray < self.experienceReplayBatchSize) then return nil end

	local experienceReplayBatch = self:sampleBatch()

	for _, experience in ipairs(experienceReplayBatch) do -- (s1, a, r, s2)

		self:update(experience[1], experience[2], experience[3], experience[4])

	end

end

function ClippedDoubleQLearningNeuralNetworkModel:reset()

	self.numberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	self.replayBufferArray = {}

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

end

function ClippedDoubleQLearningNeuralNetworkModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

	if (self.ModelParameters == nil) then self:generateLayers() end

	self.currentNumberOfEpisodes = (self.currentNumberOfEpisodes + 1) % self.maxNumberOfEpisodes

	if (self.currentNumberOfEpisodes == 0) then

		self.currentEpsilon *= self.epsilonDecayFactor

	end

	local action

	local actionVector

	local highestValue

	local highestValueVector

	local allOutputsMatrix

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < self.currentEpsilon) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		allOutputsMatrix = self:predict(currentFeatureVector, true)

		actionVector, highestValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

		action = actionVector[1][1]

		highestValue = highestValueVector[1][1]

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

	return action, highestValue

end

return ClippedDoubleQLearningNeuralNetworkModel
