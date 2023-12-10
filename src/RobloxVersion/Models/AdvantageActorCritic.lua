local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower the running reward.

function AdvantageActorCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	local NewAdvantageActorCriticModel = {}
	
	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)
	
	NewAdvantageActorCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewAdvantageActorCriticModel.epsilon = epsilon or defaultEpsilon

	NewAdvantageActorCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewAdvantageActorCriticModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewAdvantageActorCriticModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	NewAdvantageActorCriticModel.currentEpsilon = epsilon or defaultEpsilon

	NewAdvantageActorCriticModel.previousFeatureVector = nil

	NewAdvantageActorCriticModel.printReinforcementOutput = true

	NewAdvantageActorCriticModel.currentNumberOfReinforcements = 0

	NewAdvantageActorCriticModel.currentNumberOfEpisodes = 0
	
	NewAdvantageActorCriticModel.advantageHistory = {}
	
	NewAdvantageActorCriticModel.actionProbabilityHistory = {}
	
	NewAdvantageActorCriticModel.criticValueHistory = {}
	
	NewAdvantageActorCriticModel.episodeReward = 0
	
	NewAdvantageActorCriticModel.runningReward = 0
	
	NewAdvantageActorCriticModel.ClassesList = nil
	
	return NewAdvantageActorCriticModel
	
end

function AdvantageActorCriticModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.rewardAveragingRate = rewardAveragingRate or self.rewardAveragingRate
	
	self.currentEpsilon = epsilon or self.currentEpsilon
	
end

function AdvantageActorCriticModel:setActorModel(Model)
	
	self.ActorModel = Model
	
end

function AdvantageActorCriticModel:setCriticModel(Model)

	self.CriticModel = Model

end

function AdvantageActorCriticModel:setClassesList(classesList)
	
	self.ClassesList = classesList
	
end

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

local function sampleAction(actionProbabilityVector)
	
	local totalProbability = 0
	
	for _, probability in ipairs(actionProbabilityVector[1]) do
		
		totalProbability += probability
		
	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0
	
	local actionIndex = 1
	
	for i, probability in ipairs(actionProbabilityVector[1]) do
		
		cumulativeProbability += probability
		
		if (randomValue > cumulativeProbability) then continue end
			
		actionIndex = i
		
		break
		
	end
	
	return actionIndex
	
end

function AdvantageActorCriticModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local allOutputsMatrix = self.ActorModel:predict(previousFeatureVector, true)
	
	local actionProbabilityVector = calculateProbability(allOutputsMatrix)

	local previousCriticValue = self.CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local currentCriticValue = self.CriticModel:predict(currentFeatureVector, true)[1][1]
	
	local advantageValue = rewardValue + (self.discountFactor * (currentCriticValue - currentCriticValue))
	
	local numberOfActions = #allOutputsMatrix[1]
	
	local actionIndex = sampleAction(actionProbabilityVector)
	
	local action = self.ClassesList[actionIndex]
	
	local actionProbability = math.log(actionProbabilityVector[1][actionIndex])
	
	self.episodeReward += rewardValue
	
	table.insert(self.advantageHistory, advantageValue)
	
	table.insert(self.actionProbabilityHistory, actionProbability)
	
	table.insert(self.criticValueHistory, previousCriticValue)
	
	return allOutputsMatrix

end

function AdvantageActorCriticModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function AdvantageActorCriticModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function AdvantageActorCriticModel:episodeUpdate(numberOfFeatures)

	self.runningReward = (self.rewardAveragingRate * self.episodeReward) + ((1 - self.rewardAveragingRate) * self.runningReward)
	
	local historyLength = #self.advantageHistory
	
	local sumActorLosses = 0
	
	local sumCriticLosses = 0
	
	for h = 1, historyLength, 1 do
		
		local advantage = self.advantageHistory[h]
		
		local actionProbability = self.actionProbabilityHistory[h]
		
		local actorLoss = -math.log(actionProbability) * advantage
		
		local criticLoss = math.pow(advantage, 2)
		
		sumActorLosses += actorLoss
		
		sumCriticLosses += criticLoss
		
	end
	
	local lossValue = sumActorLosses + sumCriticLosses
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
	local lossVector = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList, lossValue)
	
	self.ActorModel:forwardPropagate(featureVector, true)
	self.CriticModel:forwardPropagate(featureVector, true)
	
	self.ActorModel:backPropagate(sumActorLosses, true)
	self.CriticModel:backPropagate(sumCriticLosses, true)
	
	------------------------------------------------------
	
	self.episodeReward = 0

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes += 1

	self.currentEpsilon *= self.epsilonDecayFactor
	
	table.clear(self.advantageHistory)
	
	table.clear(self.actionProbabilityHistory)
	
	table.clear(self.criticValueHistory)
	
end

function AdvantageActorCriticModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue
	
end

function AdvantageActorCriticModel:getLabelFromOutputMatrix(outputMatrix)

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

function AdvantageActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
	if (self.ActorModel == nil) then error("No actor model!") end
	
	if (self.CriticModel == nil) then error("No critic model!") end

	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then
		
		self:episodeUpdate(#currentFeatureVector[1])

	end

	self.currentNumberOfReinforcements += 1
	
	local action
	
	local actionIndex
	
	local actionVector

	local highestValue

	local highestValueVector

	local allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < self.currentEpsilon) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		if (self.previousFeatureVector) then
			
			allOutputsMatrix = self:update(self.previousFeatureVector, action, rewardValue, currentFeatureVector)
			
			actionVector, highestValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

			action = actionVector[1][1]

			highestValue = highestValueVector[1][1]
			
		end

	end

	if (self.ExperienceReplay) and (self.previousFeatureVector) then 

		self.ExperienceReplay:addExperience(self.previousFeatureVector, action, rewardValue, currentFeatureVector)

		self.ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput) then print("Episode: " .. self.currentNumberOfEpisodes .. "\t\tEpsilon: " .. self.currentEpsilon .. "\t\tReinforcement Count: " .. self.currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, highestValue
	
end

function AdvantageActorCriticModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function AdvantageActorCriticModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function AdvantageActorCriticModel:getCurrentEpsilon()

	return self.currentEpsilon

end

function AdvantageActorCriticModel:reset()
	
	self.episodeReward = 0
	
	self.runningReward = 0

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	table.clear(self.advantageHistory)
	
	table.clear(self.actionProbabilityHistory)

	table.clear(self.criticValueHistory)

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

function AdvantageActorCriticModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AdvantageActorCriticModel
