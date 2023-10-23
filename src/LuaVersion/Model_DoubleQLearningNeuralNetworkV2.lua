--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local NeuralNetworkModel = require("Model_NeuralNetwork")

DoubleQLearningNeuralNetworkModel = {}

DoubleQLearningNeuralNetworkModel.__index = DoubleQLearningNeuralNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ExperienceReplayComponent = require("Component_ExperienceReplay")

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

local function rateAverageModelParameters(averagingRate, PrimaryModelParameters, TargetModelParameters)
	
	local averagingRateComplement = 1 - averagingRate
	
	for layer = 1, #TargetModelParameters, 1 do
		
		local PrimaryModelParametersPart = AqwamMatrixLibrary:multiply(averagingRate, PrimaryModelParameters[layer])
		
		local TargetModelParametersPart = AqwamMatrixLibrary:multiply(averagingRateComplement, TargetModelParameters[layer])
		
		TargetModelParameters[layer] = AqwamMatrixLibrary:add(PrimaryModelParametersPart, TargetModelParametersPart)
		
	end
	
	return TargetModelParameters
	
end

function DoubleQLearningNeuralNetworkModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	if (self.ModelParameters == nil) then self:generateLayers() end
	
	local PrimaryModelParameters = self:getModelParameters()

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local target = rewardValue + (self.discountFactor * maxQValue[1][1])

	local targetVector = self:predict(previousFeatureVector, true)

	local actionIndex = table.find(self.ClassesList, action)

	targetVector[1][actionIndex] = target

	self:train(previousFeatureVector, targetVector)
	
	local TargetModelParameters = self:getModelParameters()
	
	TargetModelParameters = rateAverageModelParameters(self.averagingRate, PrimaryModelParameters, TargetModelParameters)
	
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
