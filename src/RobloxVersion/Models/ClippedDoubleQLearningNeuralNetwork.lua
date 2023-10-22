local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ClippedDoubleQLearningNeuralNetworkModel = {}

ClippedDoubleQLearningNeuralNetworkModel.__index = ClippedDoubleQLearningNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local ExperienceReplayComponent = require(script.Parent.Parent.Components.ExperienceReplay)

setmetatable(ClippedDoubleQLearningNeuralNetworkModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

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

	NewClippedDoubleQLearningNeuralNetworkModel.useExperienceReplay = false

	NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}

	return NewClippedDoubleQLearningNeuralNetworkModel

end

function ClippedDoubleQLearningNeuralNetworkModel:setExperienceReplay(useExperienceReplay, experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)

	self.useExperienceReplay = self:getBooleanOrDefaultOption(useExperienceReplay, self.useExperienceReplay)

	if (self.useExperienceReplay) then

		self.ExperienceReplayComponent = ExperienceReplayComponent.new(experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)

	else

		self.ExperienceReplayComponent = nil

	end

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

function ClippedDoubleQLearningNeuralNetworkModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local maxQValues = {}
	
	for i = 1, 2, 1 do
		
		self:setModelParameters(self.ModelParametersArray[i])

		local predictedValue, maxQValue = self:predict(currentFeatureVector)
		
		table.insert(maxQValues, maxQValue[1][1])
		
	end

	local maxQValue = math.min(table.unpack(maxQValues))

	local target = rewardValue + (self.discountFactor * maxQValue)

	local actionIndex = table.find(self.ClassesList, action)
	
	for i = 1, 2, 1 do
		
		self:setModelParameters(self.ModelParametersArray[i])
		
		local targetVector = self:predict(previousFeatureVector, true)
		
		targetVector[1][actionIndex] = maxQValue
		
		self:train(previousFeatureVector, targetVector)
		
	end

end

function ClippedDoubleQLearningNeuralNetworkModel:reset()

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

	if (self.useExperienceReplay) then self.ExperienceReplayComponent:reset() end

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

		self.ExperienceReplayComponent:addExperience(self.previousFeatureVector, action, rewardValue, currentFeatureVector)

		self.ExperienceReplayComponent:run(function(previousStateVector, action, rewardValue, currentStateVector)

			self:update(previousStateVector, action, rewardValue, currentStateVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput == true) then print("Current Number Of Episodes: " .. self.currentNumberOfEpisodes .. "\t\tCurrent Epsilon: " .. self.currentEpsilon) end

	if (returnOriginalOutput == true) then return allOutputsMatrix end

	return action, highestValue

end

return ClippedDoubleQLearningNeuralNetworkModel
