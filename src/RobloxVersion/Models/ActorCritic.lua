local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower the running reward.

function ActorCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	local NewActorCriticModel = {}
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	NewActorCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewActorCriticModel.epsilon = epsilon or defaultEpsilon

	NewActorCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewActorCriticModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewActorCriticModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	NewActorCriticModel.currentEpsilon = epsilon or defaultEpsilon

	NewActorCriticModel.previousFeatureVector = nil

	NewActorCriticModel.printReinforcementOutput = true

	NewActorCriticModel.currentNumberOfReinforcements = 0

	NewActorCriticModel.currentNumberOfEpisodes = 0
	
	NewActorCriticModel.actionProbabilityHistory = {}
	
	NewActorCriticModel.criticValueHistory = {}
	
	NewActorCriticModel.rewardHistory = {}
	
	NewActorCriticModel.episodeReward = 0
	
	NewActorCriticModel.runningReward = 0
	
	NewActorCriticModel.ClassesList = nil
	
	return NewActorCriticModel
	
end

function ActorCriticModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	self.currentEpsilon = epsilon or self.currentEpsilon
	
end

function ActorCriticModel:setActorModel(Model)
	
	self.ActorModel = Model
	
end

function ActorCriticModel:setCriticModel(Model)

	self.CriticModel = Model

end

function ActorCriticModel:setClassesList(classesList)
	
	self.ClassesList = classesList
	
end

local function softmax(zMatrix)

	local expMatrix = AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

	local expSum = AqwamMatrixLibrary:horizontalSum(expMatrix)

	local aMatrix = AqwamMatrixLibrary:divide(expMatrix, expSum)

	return aMatrix
	
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

function ActorCriticModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local allOutputsMatrix = self.ActorModel:predict(previousFeatureVector, true)
	
	local actionProbabilityVector = softmax(allOutputsMatrix)

	local criticValue = self.CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local numberOfActions = #allOutputsMatrix[1]
	
	local actionIndex = sampleAction(actionProbabilityVector)
	
	local action = self.ClassesList[actionIndex]
	
	local actionProbability = math.log(actionProbabilityVector[1][actionIndex])
	
	self.episodeReward += rewardValue
	
	table.insert(self.actionProbabilityHistory, actionProbability)
	
	table.insert(self.criticValueHistory, criticValue)
	
	table.insert(self.rewardHistory, rewardValue)
	
	return allOutputsMatrix

end

function ActorCriticModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function ActorCriticModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ActorCriticModel:episodeUpdate(numberOfFeatures)

	self.runningReward = (self.rewardAveragingRate * self.episodeReward) + ((1 - self.rewardAveragingRate) * self.runningReward)
	
	local returnsVector = {{}}
	
	local discountedSum = 0
	
	local historyLength = #self.rewardHistory
	
	for r = historyLength, 1, -1 do
		
		discountedSum = r + self.discountFactor * discountedSum
		
		table.insert(returnsVector[1], 1, discountedSum)
		
	end
	
	local returnsVectorMean = AqwamMatrixLibrary:mean(returnsVector)
	
	local returnsVectorStandardDeviation = AqwamMatrixLibrary:standardDeviation(returnsVector)
	
	local normalizedReturnVector = AqwamMatrixLibrary:subtract(returnsVector, returnsVectorMean)
	
	normalizedReturnVector = AqwamMatrixLibrary:divide(normalizedReturnVector, returnsVectorStandardDeviation)
	
	local sumActorLosses = 0
	
	local sumCriticLosses = 0
	
	for h = 1, historyLength, 1 do
		
		local reward = self.rewardHistory[h]
		
		local returns = normalizedReturnVector[1][h]
		
		local actionProbability = self.actionProbabilityHistory[h]
		
		local actorLoss = -math.log(actionProbability) * (returns - reward) 
		
		local criticLoss = (returns - reward)^2
		
		sumActorLosses += actorLoss
		
		sumCriticLosses += criticLoss
		
	end
	
	local lossValue = sumActorLosses + sumCriticLosses
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
	local lossVector = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList, lossValue)
	
	self.ActorModel:forwardPropagate(featureVector, true)
	self.CriticModel:forwardPropagate(featureVector, true)
	
	self.ActorModel:backPropagate(lossVector, true)
	self.CriticModel:backPropagate(lossValue, true)
	
	------------------------------------------------------
	
	self.episodeReward = 0

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes += 1

	self.currentEpsilon *= self.epsilonDecayFactor
	
	table.clear(self.actionProbabilityHistory)
	
	table.clear(self.criticValueHistory)
	
	table.clear(self.rewardHistory)
	
end

function ActorCriticModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function ActorCriticModel:getLabelFromOutputMatrix(outputMatrix)

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

function ActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
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

function ActorCriticModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function ActorCriticModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function ActorCriticModel:getCurrentEpsilon()

	return self.currentEpsilon

end

function ActorCriticModel:reset()
	
	self.episodeReward = 0
	
	self.runningReward = 0

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	table.clear(self.actionProbabilityHistory)

	table.clear(self.criticValueHistory)

	table.clear(self.rewardHistory)

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

function ActorCriticModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ActorCriticModel
