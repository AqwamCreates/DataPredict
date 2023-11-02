local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower running reward

function ActorCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	local NewActorCriticModel = {}
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	NewActorCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewActorCriticModel.epsilon = epsilon or defaultEpsilon

	NewActorCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewActorCriticModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewActorCriticModel.currentEpsilon = epsilon or defaultEpsilon
	
	NewActorCriticModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate

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

	local allOutputsVector = self.ActorModel:predict(previousFeatureVector, true)
	
	local actionProbabilityVector = softmax(allOutputsVector)

	local criticValue = self.CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local numberOfActions = #allOutputsVector[1]
	
	local actionIndex = sampleAction(actionProbabilityVector)
	
	local action = self.ClassesList[actionIndex]
	
	local actionProbability = math.log(actionProbabilityVector[1][actionIndex])
	
	self.episodeReward += rewardValue
	
	table.insert(self.actionProbabilityHistory, actionProbability)
	
	table.insert(self.criticValueHistory, criticValue)
	
	table.insert(self.rewardHistory, rewardValue)
	
	return action, actionProbabilityVector, actionIndex

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

function ActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
	if (self.ActorModel == nil) then error("No actor model!") end
	
	if (self.CriticModel == nil) then error("No critic model!") end

	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then
		
		self:episodeUpdate(#currentFeatureVector[1])

	end

	self.currentNumberOfReinforcements += 1
	
	local action
	
	local actionIndex

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
			
			action, highestValueVector, actionIndex = self:update(self.previousFeatureVector, action, rewardValue, currentFeatureVector)

			highestValue = highestValueVector[1][1]
			
			allOutputsMatrix[1][actionIndex] = highestValue
			
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

return ActorCriticModel
