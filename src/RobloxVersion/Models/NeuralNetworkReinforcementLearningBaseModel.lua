local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

NeuralNetworkReinforcementLearningBaseModel = {}

NeuralNetworkReinforcementLearningBaseModel.__index = NeuralNetworkReinforcementLearningBaseModel

setmetatable(NeuralNetworkReinforcementLearningBaseModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

function NeuralNetworkReinforcementLearningBaseModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewNeuralNetworkReinforcementLearningBaseModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	NewNeuralNetworkReinforcementLearningBaseModel:setPrintOutput(false)

	setmetatable(NewNeuralNetworkReinforcementLearningBaseModel, NeuralNetworkReinforcementLearningBaseModel)

	NewNeuralNetworkReinforcementLearningBaseModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewNeuralNetworkReinforcementLearningBaseModel.epsilon = epsilon or defaultEpsilon

	NewNeuralNetworkReinforcementLearningBaseModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewNeuralNetworkReinforcementLearningBaseModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewNeuralNetworkReinforcementLearningBaseModel.currentNumberOfEpisodes = 0

	NewNeuralNetworkReinforcementLearningBaseModel.currentEpsilon = epsilon or defaultEpsilon

	NewNeuralNetworkReinforcementLearningBaseModel.previousFeatureVector = nil

	NewNeuralNetworkReinforcementLearningBaseModel.printReinforcementOutput = true

	return NewNeuralNetworkReinforcementLearningBaseModel
	
end

function NeuralNetworkReinforcementLearningBaseModel:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function NeuralNetworkReinforcementLearningBaseModel:setUpdateFunction(updateFunction)
	
	self.updateFunction = updateFunction
	
end

function NeuralNetworkReinforcementLearningBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
end

function NeuralNetworkReinforcementLearningBaseModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function NeuralNetworkReinforcementLearningBaseModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function NeuralNetworkReinforcementLearningBaseModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

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

	if (self.ExperienceReplay) and (self.previousFeatureVector) then 

		self.ExperienceReplay:addExperience(self.previousFeatureVector, action, rewardValue, currentFeatureVector)

		self.ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput == true) then print("Current Number Of Episodes: " .. self.currentNumberOfEpisodes .. "\t\tCurrent Epsilon: " .. self.currentEpsilon) end

	if (returnOriginalOutput == true) then return allOutputsMatrix end

	return action, highestValue

end

function NeuralNetworkReinforcementLearningBaseModel:reset()

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

return NeuralNetworkReinforcementLearningBaseModel
