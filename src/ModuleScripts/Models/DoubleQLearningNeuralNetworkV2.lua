local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

DoubleQLearningNeuralNetworkModel = {}

DoubleQLearningNeuralNetworkModel.__index = DoubleQLearningNeuralNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local ExperienceReplayComponent = require(script.Parent.Parent.Components.ExperienceReplay)

setmetatable(DoubleQLearningNeuralNetworkModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultAveragingRate = 0.01

local defaultMaxNumberOfIterations = 1

function DoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor, averagingRate)

	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewDoubleQLearningNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	NewDoubleQLearningNeuralNetworkModel:setPrintOutput(false)

	setmetatable(NewDoubleQLearningNeuralNetworkModel, DoubleQLearningNeuralNetworkModel)

	NewDoubleQLearningNeuralNetworkModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewDoubleQLearningNeuralNetworkModel.epsilon = epsilon or defaultEpsilon

	NewDoubleQLearningNeuralNetworkModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewDoubleQLearningNeuralNetworkModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewDoubleQLearningNeuralNetworkModel.averagingRate = averagingRate or defaultAveragingRate

	NewDoubleQLearningNeuralNetworkModel.currentNumberOfEpisodes = 0

	NewDoubleQLearningNeuralNetworkModel.currentEpsilon = epsilon or defaultEpsilon

	NewDoubleQLearningNeuralNetworkModel.previousFeatureVector = nil

	NewDoubleQLearningNeuralNetworkModel.printReinforcementOutput = true

	NewDoubleQLearningNeuralNetworkModel.useExperienceReplay = false
	
	NewDoubleQLearningNeuralNetworkModel.PrimaryModelParameters = nil
	
	NewDoubleQLearningNeuralNetworkModel.ExperienceReplayComponent = nil

	return NewDoubleQLearningNeuralNetworkModel

end

function DoubleQLearningNeuralNetworkModel:setExperienceReplay(useExperienceReplay, experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)

	self.useExperienceReplay = self:getBooleanOrDefaultOption(useExperienceReplay, self.useExperienceReplay)
	
	if (self.useExperienceReplay) then
		
		self.ExperienceReplayComponent = ExperienceReplayComponent.new(experienceReplayBatchSize, numberOfReinforcementsForExperienceReplayUpdate, maxExperienceReplayBufferSize)
		
	else
		
		self.ExperienceReplayComponent = nil
		
	end

end

function DoubleQLearningNeuralNetworkModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function DoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor, averagingRate)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor
	
	self.averagingRate = averagingRate or self.averagingRate

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function DoubleQLearningNeuralNetworkModel:rateAverageModelParameters()
	
	local TargetModelParameters = self:getModelParameters()
	
	for layer = 1, #TargetModelParameters, 1 do
		
		local PrimaryModelParametersPart = AqwamMatrixLibrary:multiply(self.averagingRate, self.PrimaryModelParameters[layer])
		
		local TargetModelParametersPart = AqwamMatrixLibrary:multiply((1 - self.averagingRate), TargetModelParameters[layer])
		
		TargetModelParameters[layer] = AqwamMatrixLibrary:add(PrimaryModelParametersPart, TargetModelParametersPart)
		
	end
	
	return TargetModelParameters
	
end

function DoubleQLearningNeuralNetworkModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	if (self.PrimaryModelParameters == nil) then self:generateLayers() end
	
	self.PrimaryModelParameters = self:getModelParameters()

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local target = rewardValue + (self.discountFactor * maxQValue[1][1])

	local targetVector = self:predict(previousFeatureVector, true)

	local actionIndex = table.find(self.ClassesList, action)

	targetVector[1][actionIndex] = target

	self:train(previousFeatureVector, targetVector)
	
	local TargetModelParameters = self:rateAverageModelParameters()
	
	self:setModelParameters(TargetModelParameters)

end

function DoubleQLearningNeuralNetworkModel:reset()

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end
	
	if (self.useExperienceReplay) then self.ExperienceReplayComponent:reset() end

end

function DoubleQLearningNeuralNetworkModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

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

		self.ExperienceReplayComponent:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)
		
	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput == true) then print("Current Number Of Episodes: " .. self.currentNumberOfEpisodes .. "\t\tCurrent Epsilon: " .. self.currentEpsilon) end

	if (returnOriginalOutput == true) then return allOutputsMatrix end

	return action, highestValue

end

return DoubleQLearningNeuralNetworkModel
