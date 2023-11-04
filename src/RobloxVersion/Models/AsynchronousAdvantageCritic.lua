local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

AsynchronousAdvantageCriticModel = {}

AsynchronousAdvantageCriticModel.__index = AsynchronousAdvantageCriticModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower the running reward.

local defaultTotalNumberOfReinforcementsToUpdateMainModel

function AsynchronousAdvantageCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate, totalNumberOfReinforcementsToUpdateMainModel)
	
	local NewAsynchronousAdvantageCriticModel = {}
	
	setmetatable(NewAsynchronousAdvantageCriticModel, AsynchronousAdvantageCriticModel)
	
	NewAsynchronousAdvantageCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewAsynchronousAdvantageCriticModel.epsilon = epsilon or defaultEpsilon

	NewAsynchronousAdvantageCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewAsynchronousAdvantageCriticModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewAsynchronousAdvantageCriticModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	NewAsynchronousAdvantageCriticModel.currentEpsilonArray = {}

	NewAsynchronousAdvantageCriticModel.previousFeatureVectorArray = {}

	NewAsynchronousAdvantageCriticModel.printReinforcementOutput = true

	NewAsynchronousAdvantageCriticModel.currentNumberOfReinforcementsArray = {}

	NewAsynchronousAdvantageCriticModel.currentNumberOfEpisodesArray = {}
	
	NewAsynchronousAdvantageCriticModel.advantageHistoryArray = {}
	
	NewAsynchronousAdvantageCriticModel.actionProbabilityHistoryArray = {}
	
	NewAsynchronousAdvantageCriticModel.criticValueHistoryArray = {}
	
	NewAsynchronousAdvantageCriticModel.episodeRewardArray = {}
	
	NewAsynchronousAdvantageCriticModel.runningRewardArray = {}
	
	NewAsynchronousAdvantageCriticModel.ActorModelArray = {}
	
	NewAsynchronousAdvantageCriticModel.CriticModelArray = {}
	
	NewAsynchronousAdvantageCriticModel.ExperienceReplayArray = {}
	
	NewAsynchronousAdvantageCriticModel.ClassesList = nil
	
	NewAsynchronousAdvantageCriticModel.totalNumberOfReinforcementsToUpdateMainModel = totalNumberOfReinforcementsToUpdateMainModel or 0
	
	NewAsynchronousAdvantageCriticModel.currentTotalNumberOfReinforcementsToUpdateMainModel = 0
	
	return NewAsynchronousAdvantageCriticModel
	
end

function AsynchronousAdvantageCriticModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	for i = 1, #self.previousFeatureVectorArray, 1 do
		
		self.currentEpsilon[i] = epsilon or self.currentEpsilon[i]
		
	end
	
end

function AsynchronousAdvantageCriticModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function AsynchronousAdvantageCriticModel:addActorCriticModel(ActorModel, CriticModel, ExperienceReplay)
	
	if not ActorModel then error("No actor model!") end
	
	if not CriticModel then error("No critic model!") end
	
	table.insert(self.ActorModelArray, ActorModel)
		
	table.insert(self.CriticModelArray, CriticModel)
	
	if ExperienceReplay then table.insert(self.ExperienceReplayArray, ExperienceReplay) end
	
	table.insert(self.episodeRewardArray,  0)

	table.insert(self.currentNumberOfReinforcementsArray,  0)

	table.insert(self.currentNumberOfEpisodesArray,  0)

	table.insert(self.currentEpsilonArray,  0)

	table.insert(self.runningRewardArray,  0)

	table.insert(self.advantageHistoryArray, {})

	table.insert(self.actionProbabilityHistoryArray, {})

	table.insert(self.criticValueHistoryArray, {})
	
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

function AsynchronousAdvantageCriticModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber)
	
	local ActorModel = self.ActorModelArray[actorCriticModelNumber]
	
	local CriticModel = self.CriticModelArray[actorCriticModelNumber]

	local allOutputsMatrix = ActorModel:predict(previousFeatureVector, true)
	
	local actionProbabilityVector = softmax(allOutputsMatrix)

	local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]
	
	local advantageValue = rewardValue + (self.discountFactor * (currentCriticValue - currentCriticValue))
	
	local numberOfActions = #allOutputsMatrix[1]
	
	local actionIndex = sampleAction(actionProbabilityVector)
	
	local action = self.ClassesList[actionIndex]
	
	local actionProbability = math.log(actionProbabilityVector[1][actionIndex])
	
	self.episodeRewardArray[actorCriticModelNumber] += rewardValue
	
	table.insert(self.advantageHistoryArray[actorCriticModelNumber], advantageValue)
	
	table.insert(self.actionProbabilityHistoryArray[actorCriticModelNumber], actionProbability)
	
	table.insert(self.criticValueHistoryArray[actorCriticModelNumber], previousCriticValue)
	
	return allOutputsMatrix

end

function AsynchronousAdvantageCriticModel:episodeUpdate(numberOfFeatures, actorCriticModelNumber)

	self.runningRewardArray[actorCriticModelNumber] = (self.rewardAveragingRate * self.episodeRewardArray[actorCriticModelNumber]) + ((1 - self.rewardAveragingRate) * self.runningRewardArray[actorCriticModelNumber])
	
	local historyLength = #self.advantageHistoryArray[actorCriticModelNumber]
	
	local sumActorLosses = 0
	
	local sumCriticLosses = 0
	
	for h = 1, historyLength, 1 do
		
		local advantage = self.advantageHistoryArray[actorCriticModelNumber][h]
		
		local actionProbability = self.actionProbabilityHistoryArray[actorCriticModelNumber][h]
		
		local actorLoss = -math.log(actionProbability) * advantage
		
		local criticLoss = math.pow(advantage, 2)
		
		sumActorLosses += actorLoss
		
		sumCriticLosses += criticLoss
		
	end
	
	local lossValue = sumActorLosses + sumCriticLosses
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
	local lossVector = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList, lossValue)
	
	local ActorModel = self.ActorModelArray[actorCriticModelNumber]
	local CriticModel = self.CriticModelArray[actorCriticModelNumber]
	
	ActorModel:forwardPropagate(featureVector, true)
	CriticModel:forwardPropagate(featureVector, true)
	
	ActorModel:backPropagate(sumActorLosses, true)
	CriticModel:backPropagate(sumCriticLosses, true)
	
	------------------------------------------------------
	
	self.episodeRewardArray[actorCriticModelNumber] = 0

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] += 1

	self.currentEpsilonArray[actorCriticModelNumber] *= self.epsilonDecayFactor
	
	table.clear(self.advantageHistoryArray[actorCriticModelNumber])
	
	table.clear(self.actionProbabilityHistoryArray[actorCriticModelNumber])
	
	table.clear(self.criticValueHistoryArray[actorCriticModelNumber])
	
end

function AsynchronousAdvantageCriticModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue
	
end

function AsynchronousAdvantageCriticModel:getLabelFromOutputMatrix(outputMatrix)

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

function AsynchronousAdvantageCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, actorCriticModelNumber)

	if (self.currentNumberOfReinforcementsArray[actorCriticModelNumber] >= self.numberOfReinforcementsPerEpisode) then
		
		self:episodeUpdate(#currentFeatureVector[1], actorCriticModelNumber)

	end

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] += 1
	
	self.currentTotalNumberOfReinforcementsToUpdateMainModel += 1
	
	local action
	
	local actionIndex
	
	local actionVector

	local highestValue

	local highestValueVector

	local allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)

	local randomProbability = Random.new():NextNumber()
	
	local previousFeatureVector = self.previousFeatureVectorArray[actorCriticModelNumber]
	
	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	if (randomProbability < self.currentEpsilonArray[actorCriticModelNumber]) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		if (previousFeatureVector) then
			
			allOutputsMatrix = self:update(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber)
			
			actionVector, highestValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

			action = actionVector[1][1]

			highestValue = highestValueVector[1][1]
			
		end

	end

	if (ExperienceReplay) and (previousFeatureVector) then 

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, actorCriticModelNumber)

		end)

	end

	self.previousFeatureVectorArray[actorCriticModelNumber] = currentFeatureVector

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, highestValue
	
end

function AsynchronousAdvantageCriticModel:getCurrentNumberOfEpisodes(actorCriticModelNumber)

	return self.currentNumberOfEpisodesArray[actorCriticModelNumber]

end

function AsynchronousAdvantageCriticModel:getCurrentNumberOfReinforcements(actorCriticModelNumber)

	return self.currentNumberOfReinforcementsArray[actorCriticModelNumber]

end

function AsynchronousAdvantageCriticModel:getCurrentEpsilon(actorCriticModelNumber)

	return self.currentEpsilonArray[actorCriticModelNumber]

end

function AsynchronousAdvantageCriticModel:singleReset(actorCriticModelNumber)
	
	self.episodeRewardArray[actorCriticModelNumber] = 0
	
	self.runningRewardArray[actorCriticModelNumber] = 0

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] = 0

	self.previousFeatureVectorArray[actorCriticModelNumber] = nil

	self.currentEpsilonArray[actorCriticModelNumber] = self.epsilon
	
	table.clear(self.advantageHistoryArray[actorCriticModelNumber])

	table.clear(self.actionProbabilityHistoryArray[actorCriticModelNumber])

	table.clear(self.criticValueHistoryArray[actorCriticModelNumber])
	
	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	if (ExperienceReplay) then ExperienceReplay:reset() end

end

function AsynchronousAdvantageCriticModel:reset()
	
	for i = 1, #self.currentEpsilonArray, 1 do self:singleReset(i) end
	
	self.currentTotalNumberOfReinforcementsToUpdateMainModel = 0
	
end

function AsynchronousAdvantageCriticModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AsynchronousAdvantageCriticModel
