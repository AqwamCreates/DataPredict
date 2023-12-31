local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

function ProximalPolicyOptimizationModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	local NewProximalPolicyOptimizationModel = {}
	
	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)
	
	NewProximalPolicyOptimizationModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewProximalPolicyOptimizationModel.epsilon = epsilon or defaultEpsilon

	NewProximalPolicyOptimizationModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewProximalPolicyOptimizationModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewProximalPolicyOptimizationModel.currentEpsilon = epsilon or defaultEpsilon

	NewProximalPolicyOptimizationModel.previousFeatureVector = nil

	NewProximalPolicyOptimizationModel.printReinforcementOutput = true

	NewProximalPolicyOptimizationModel.currentNumberOfReinforcements = 0

	NewProximalPolicyOptimizationModel.currentNumberOfEpisodes = 0
	
	NewProximalPolicyOptimizationModel.advantageHistory = {}
	
	NewProximalPolicyOptimizationModel.actionVectorHistory = {}
	
	NewProximalPolicyOptimizationModel.ClassesList = nil
	
	return NewProximalPolicyOptimizationModel
	
end

function ProximalPolicyOptimizationModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor
	
	self.currentEpsilon = epsilon or self.currentEpsilon
	
end

function ProximalPolicyOptimizationModel:setActorModel(Model)
	
	self.ActorModel = Model
	
end

function ProximalPolicyOptimizationModel:setCriticModel(Model)

	self.CriticModel = Model

end

function ProximalPolicyOptimizationModel:setClassesList(classesList)
	
	self.ClassesList = classesList
	
end

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

function ProximalPolicyOptimizationModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local allOutputsMatrix = self.ActorModel:predict(previousFeatureVector, true)
	
	local actionProbabilityVector = calculateProbability(allOutputsMatrix)

	local previousCriticValue = self.CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local currentCriticValue = self.CriticModel:predict(currentFeatureVector, true)[1][1]
	
	local advantageValue = rewardValue + (self.discountFactor * (currentCriticValue - previousCriticValue))
	
	table.insert(self.advantageHistory, advantageValue)
	
	table.insert(self.actionVectorHistory, actionProbabilityVector)
	
	return allOutputsMatrix

end

function ProximalPolicyOptimizationModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

local function convertListOfVectorsToMatrix(listOfVectors)
	
	local matrix = {}
	
	for i = 1, #listOfVectors, 1 do
		
		table.insert(matrix, listOfVectors[i][1])
		
	end
	
	return matrix
	
end

function ProximalPolicyOptimizationModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ProximalPolicyOptimizationModel:episodeUpdate(numberOfFeatures)
	
	local historyLength = #self.advantageHistory
	
	local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)
	
	local sumCriticLosses = 0
	
	local actionMatrix = convertListOfVectorsToMatrix(self.actionVectorHistory)
	
	for h = 1, historyLength, 1 do
		
		local advantage = self.advantageHistory[h]
		
		local currentActionVector = {actionMatrix[h]}
		
		local ratioVector = AqwamMatrixLibrary:divide(currentActionVector, actionMatrix)
		
		local actorLossVector = AqwamMatrixLibrary:multiply(-1, ratioVector, advantage)
		
		sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)
		
		sumCriticLosses += advantage
		
	end
	
	local featureVector = AqwamMatrixLibrary:createMatrix(historyLength, numberOfFeatures, 1)
	
	self.ActorModel:forwardPropagate(featureVector, true)
	self.CriticModel:forwardPropagate(featureVector, true)
	
	self.ActorModel:backPropagate(sumActorLossVector, true)
	self.CriticModel:backPropagate(sumCriticLosses, true)
	
	------------------------------------------------------
	
	self.episodeReward = 0

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes += 1

	self.currentEpsilon *= self.epsilonDecayFactor
	
	table.clear(self.advantageHistory)
	
	table.clear(self.actionVectorHistory)
	
end

function ProximalPolicyOptimizationModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue
	
end

function ProximalPolicyOptimizationModel:getLabelFromOutputMatrix(outputMatrix)

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

function ProximalPolicyOptimizationModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
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

function ProximalPolicyOptimizationModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function ProximalPolicyOptimizationModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function ProximalPolicyOptimizationModel:getCurrentEpsilon()

	return self.currentEpsilon

end

function ProximalPolicyOptimizationModel:reset()

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	table.clear(self.advantageHistory)
	
	table.clear(self.actionVectorHistory)

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

function ProximalPolicyOptimizationModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ProximalPolicyOptimizationModel
