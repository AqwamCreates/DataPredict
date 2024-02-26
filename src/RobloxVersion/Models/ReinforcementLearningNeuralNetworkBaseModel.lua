local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ReinforcementLearningNeuralNetworkBaseModel = {}

ReinforcementLearningNeuralNetworkBaseModel.__index = ReinforcementLearningNeuralNetworkBaseModel

setmetatable(ReinforcementLearningNeuralNetworkBaseModel, NeuralNetworkModel)

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

function ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewReinforcementLearningNeuralNetworkBaseModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate)

	NewReinforcementLearningNeuralNetworkBaseModel:setPrintOutput(false)

	setmetatable(NewReinforcementLearningNeuralNetworkBaseModel, ReinforcementLearningNeuralNetworkBaseModel)

	NewReinforcementLearningNeuralNetworkBaseModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewReinforcementLearningNeuralNetworkBaseModel.epsilon = epsilon or defaultEpsilon

	NewReinforcementLearningNeuralNetworkBaseModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewReinforcementLearningNeuralNetworkBaseModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewReinforcementLearningNeuralNetworkBaseModel.currentEpsilon = epsilon or defaultEpsilon

	NewReinforcementLearningNeuralNetworkBaseModel.previousFeatureVector = nil

	NewReinforcementLearningNeuralNetworkBaseModel.printReinforcementOutput = true
	
	NewReinforcementLearningNeuralNetworkBaseModel.currentNumberOfReinforcements = 0
	
	NewReinforcementLearningNeuralNetworkBaseModel.currentNumberOfEpisodes = 0

	return NewReinforcementLearningNeuralNetworkBaseModel
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function ReinforcementLearningNeuralNetworkBaseModel:setUpdateFunction(updateFunction)
	
	self.updateFunction = updateFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
	
	self.episodeUpdateFunction = episodeUpdateFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	return self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
end

function ReinforcementLearningNeuralNetworkBaseModel:episodeUpdate()
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes += 1

	self.currentEpsilon *= self.epsilonDecayFactor
	
	if not episodeUpdateFunction then return end
	
	episodeUpdateFunction()
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function ReinforcementLearningNeuralNetworkBaseModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = self:getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ReinforcementLearningNeuralNetworkBaseModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

	if (self.ModelParameters == nil) then self:generateLayers() end
	
	self.currentNumberOfReinforcements += 1

	local action

	local actionVector

	local highestValue

	local highestValueVector

	local allOutputsMatrix
	
	local temporalDifferenceError

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < self.currentEpsilon) then
		
		local numberOfClasses = #self.ClassesList

		local randomNumber = Random.new():NextInteger(1, numberOfClasses)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		allOutputsMatrix = self:predict(currentFeatureVector, true)

		actionVector, highestValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

		action = actionVector[1][1]

		highestValue = highestValueVector[1][1]

	end

	if (self.previousFeatureVector) then 
		
		temporalDifferenceError = self:update(self.previousFeatureVector, action, rewardValue, currentFeatureVector) 
		
	end
	
	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then

		self:episodeUpdate()

	end

	if (self.ExperienceReplay) and (self.previousFeatureVector) then 

		self.ExperienceReplay:addExperience(self.previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		self.ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

		self.ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput) then print("Episode: " .. self.currentNumberOfEpisodes .. "\t\tEpsilon: " .. self.currentEpsilon .. "\t\tReinforcement Count: " .. self.currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, highestValue

end

function ReinforcementLearningNeuralNetworkBaseModel:getCurrentNumberOfEpisodes()
	
	return self.currentNumberOfEpisodes
	
end

function ReinforcementLearningNeuralNetworkBaseModel:getCurrentNumberOfReinforcements()
	
	return self.currentNumberOfReinforcements
	
end

function ReinforcementLearningNeuralNetworkBaseModel:getCurrentEpsilon()
	
	return self.currentEpsilon
	
end

function ReinforcementLearningNeuralNetworkBaseModel:extendResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:reset()
	
	self.currentNumberOfReinforcements = 0
	
	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end
	
	if (self.resetFunction) then self.resetFunction() end

end

return ReinforcementLearningNeuralNetworkBaseModel
