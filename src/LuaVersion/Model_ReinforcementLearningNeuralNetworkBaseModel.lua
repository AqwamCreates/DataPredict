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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local NeuralNetworkModel = require("Model_NeuralNetwork")

ReinforcementLearningNeuralNetworkBaseModel = {}

ReinforcementLearningNeuralNetworkBaseModel.__index = ReinforcementLearningNeuralNetworkBaseModel

setmetatable(ReinforcementLearningNeuralNetworkBaseModel, NeuralNetworkModel)

local defaultMaxNumberOfEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

function ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewReinforcementLearningNeuralNetworkBaseModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	NewReinforcementLearningNeuralNetworkBaseModel:setPrintOutput(false)

	setmetatable(NewReinforcementLearningNeuralNetworkBaseModel, ReinforcementLearningNeuralNetworkBaseModel)

	NewReinforcementLearningNeuralNetworkBaseModel.maxNumberOfEpisodes = maxNumberOfEpisodes or defaultMaxNumberOfEpisode

	NewReinforcementLearningNeuralNetworkBaseModel.epsilon = epsilon or defaultEpsilon

	NewReinforcementLearningNeuralNetworkBaseModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewReinforcementLearningNeuralNetworkBaseModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewReinforcementLearningNeuralNetworkBaseModel.currentNumberOfEpisodes = 0

	NewReinforcementLearningNeuralNetworkBaseModel.currentEpsilon = epsilon or defaultEpsilon

	NewReinforcementLearningNeuralNetworkBaseModel.previousFeatureVector = nil

	NewReinforcementLearningNeuralNetworkBaseModel.printReinforcementOutput = true

	return NewReinforcementLearningNeuralNetworkBaseModel
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setParameters(maxNumberOfIterations, learningRate, targetCost, maxNumberOfEpisodes, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.maxNumberOfEpisodes = maxNumberOfEpisodes or self.maxNumberOfEpisodes

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function ReinforcementLearningNeuralNetworkBaseModel:setUpdateFunction(updateFunction)
	
	self.updateFunction = updateFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function ReinforcementLearningNeuralNetworkBaseModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ReinforcementLearningNeuralNetworkBaseModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

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

function ReinforcementLearningNeuralNetworkBaseModel:reset()

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

return ReinforcementLearningNeuralNetworkBaseModel
