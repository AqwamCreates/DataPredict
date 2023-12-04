local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

DuelingQLearningModel = {}

DuelingQLearningModel.__index = DuelingQLearningModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower the running reward.

function DuelingQLearningModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)

	local NewDuelingQLearningModel = {}

	setmetatable(NewDuelingQLearningModel, DuelingQLearningModel)

	NewDuelingQLearningModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewDuelingQLearningModel.epsilon = epsilon or defaultEpsilon

	NewDuelingQLearningModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewDuelingQLearningModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewDuelingQLearningModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate

	NewDuelingQLearningModel.currentEpsilon = epsilon or defaultEpsilon

	NewDuelingQLearningModel.previousFeatureVector = nil

	NewDuelingQLearningModel.printReinforcementOutput = true

	NewDuelingQLearningModel.currentNumberOfReinforcements = 0

	NewDuelingQLearningModel.currentNumberOfEpisodes = 0

	NewDuelingQLearningModel.ClassesList = nil

	return NewDuelingQLearningModel

end

function DuelingQLearningModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.rewardAveragingRate = rewardAveragingRate or self.rewardAveragingRate

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function DuelingQLearningModel:setAdvantageModel(Model)

	self.AdvantageModel = Model

end

function DuelingQLearningModel:setValueModel(Model)

	self.ValueModel = Model

end

function DuelingQLearningModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function DuelingQLearningModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function DuelingQLearningModel:getLabelFromOutputMatrix(outputMatrix)

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

function DuelingQLearningModel:forwardPropagate(featureVector)

	local value = self.ValueModel:predict(featureVector, true)[1][1]

	local advantageMatrix = self.AdvantageModel:predict(featureVector, true)

	local meanAdvantageVector = AqwamMatrixLibrary:horizontalMean(advantageMatrix)

	local qValuePart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValue = AqwamMatrixLibrary:add(value, qValuePart1)

	return qValue, value

end

function DuelingQLearningModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local previousQValue, previousValue = self:forwardPropagate(previousFeatureVector)

	local currentQValue, currentValue = self:forwardPropagate(currentFeatureVector)

	local _, maxCurrentQValue = self:fetchHighestValueInVector(currentQValue)

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLoss = AqwamMatrixLibrary:subtract(previousQValue, expectedQValue)

	local vLoss = AqwamMatrixLibrary:subtract(previousValue, currentValue)

	self.ValueModel:forwardPropagate(previousFeatureVector, true)

	self.ValueModel:backPropagate(qLoss, true)

	local allOutputsMatrix = self.AdvantageModel:forwardPropagate(previousFeatureVector, true)

	self.AdvantageModel:backPropagate(vLoss, true)

	return allOutputsMatrix

end

function DuelingQLearningModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function DuelingQLearningModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function DuelingQLearningModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

	if (self.ValueModel == nil) then error("No value model!") end

	if (self.AdvantageModel == nil) then error("No advantage model!") end

	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then

		self.currentNumberOfReinforcements = 0

		self.currentNumberOfEpisodes += 1

		self.currentEpsilon *= self.epsilonDecayFactor

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

function DuelingQLearningModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function DuelingQLearningModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function DuelingQLearningModel:getCurrentEpsilon()

	return self.currentEpsilon

end

function DuelingQLearningModel:reset()

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

function DuelingQLearningModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DuelingQLearningModel
