local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ReinforcementLearningActorCriticNeuralNetworkBaseModel = {}

ReinforcementLearningActorCriticNeuralNetworkBaseModel.__index = ReinforcementLearningActorCriticNeuralNetworkBaseModel

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

function ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	local NewReinforcementLearningActorCriticNeuralNetworkBaseModel = {}
	
	setmetatable(NewReinforcementLearningActorCriticNeuralNetworkBaseModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)
	
	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode
	
	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.epsilon = epsilon or defaultEpsilon
	
	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.epsilonDecayFactor = epsilonDecayFactor or defaultEpsilon
	
	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.discountFactor = discountFactor or defaultDiscountFactor
	
	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.currentEpsilon = epsilon or defaultEpsilon

	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.previousFeatureVector = nil

	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.printReinforcementOutput = true

	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.currentNumberOfReinforcements = 0

	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.currentNumberOfEpisodes = 0
	
	return NewReinforcementLearningActorCriticNeuralNetworkBaseModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setActorModel(ActorModel)
	
	ActorModel:setPrintOutput(false)
	
	self.ActorModel = ActorModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setCriticModel(CriticModel)
	
	CriticModel:setPrintOutput(false)

	self.CriticModel = CriticModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:episodeUpdate()

	local episodeUpdateFunction = self.episodeUpdateFunction
	
	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes += 1

	self.currentEpsilon *= self.epsilonDecayFactor

	if not episodeUpdateFunction then return end

	episodeUpdateFunction()

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getLabelFromOutputMatrix(outputMatrix)

	local predictedLabelVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValueVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValue

	local outputVector

	local classIndex

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		predictedLabel, highestValue = self:fetchHighestValueInVector(outputVector)

		predictedLabelVector[i][1] = predictedLabel

		highestValueVector[i][1] = highestValue

	end

	return predictedLabelVector, highestValueVector

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
	if (self.ActorModel == nil) then error("No actor model!") end

	if (self.CriticModel == nil) then error("No critic model!") end

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

		allOutputsMatrix = self.ActorModel:predict(currentFeatureVector, true)

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

			return self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput) then print("Episode: " .. self.currentNumberOfEpisodes .. "\t\tEpsilon: " .. self.currentEpsilon .. "\t\tReinforcement Count: " .. self.currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, highestValue

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getActorModel()
	
	return self.ActorModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getCriticModel()

	return self.CriticModel

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getCurrentEpsilon()

	return self.currentEpsilon

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:extendResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:reset()

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	local ActorModel = self.ActorModel
	
	local CriticModel = self.CriticModel
	
	if (ActorModel) then ActorModel:reset() end
	
	if (CriticModel) then CriticModel:reset() end

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

	if (self.resetFunction) then self.resetFunction() end

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningActorCriticNeuralNetworkBaseModel
