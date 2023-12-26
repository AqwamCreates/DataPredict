local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

VanillaPolicyGradientModel = {}

VanillaPolicyGradientModel.__index = VanillaPolicyGradientModel

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower the running reward.

function VanillaPolicyGradientModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate)
	
	local NewVanillaPolicyGradientModel = {}
	
	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)
	
	NewVanillaPolicyGradientModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewVanillaPolicyGradientModel.epsilon = epsilon or defaultEpsilon

	NewVanillaPolicyGradientModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewVanillaPolicyGradientModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewVanillaPolicyGradientModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	NewVanillaPolicyGradientModel.currentEpsilon = epsilon or defaultEpsilon

	NewVanillaPolicyGradientModel.previousFeatureVector = nil

	NewVanillaPolicyGradientModel.printReinforcementOutput = true

	NewVanillaPolicyGradientModel.currentNumberOfReinforcements = 0

	NewVanillaPolicyGradientModel.currentNumberOfEpisodes = 0
	
	NewVanillaPolicyGradientModel.advantageHistory = {}
	
	NewVanillaPolicyGradientModel.gradientHistory = {}
	
	NewVanillaPolicyGradientModel.ClassesList = nil
	
	return NewVanillaPolicyGradientModel
	
end

function VanillaPolicyGradientModel:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon
	
end

function VanillaPolicyGradientModel:setActorModel(Model)
	
	self.ActorModel = Model
	
end

function VanillaPolicyGradientModel:setCriticModel(Model)

	self.CriticModel = Model

end

function VanillaPolicyGradientModel:setClassesList(classesList)
	
	self.ClassesList = classesList
	
end

function VanillaPolicyGradientModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local allOutputsMatrix = self.ActorModel:predict(previousFeatureVector, true)
	
	local logOutputMatrix = AqwamMatrixLibrary:applyFunction(math.log, allOutputsMatrix)

	local previousCriticValue = self.CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local currentCriticValue = self.CriticModel:predict(currentFeatureVector, true)[1][1]
	
	local advantageValue = rewardValue + (self.discountFactor * (currentCriticValue - previousCriticValue))
	
	local gradientMatrix = AqwamMatrixLibrary:multiply(logOutputMatrix, advantageValue)
	
	table.insert(self.gradientHistory, gradientMatrix[1])
	
	table.insert(self.advantageHistory, {advantageValue})
	
	return allOutputsMatrix

end

function VanillaPolicyGradientModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function VanillaPolicyGradientModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function VanillaPolicyGradientModel:episodeUpdate(numberOfFeatures)
	
	local sumGradient = AqwamMatrixLibrary:verticalSum(self.gradientHistory)
	
	local sumAdvantage = AqwamMatrixLibrary:verticalSum(self.advantageHistory)
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
	
	self.ActorModel:forwardPropagate(featureVector, true)
	self.CriticModel:forwardPropagate(featureVector, true)
	
	self.ActorModel:backPropagate(sumGradient, true)
	self.CriticModel:backPropagate(sumAdvantage, true)
	
	------------------------------------------------------

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes += 1

	self.currentEpsilon *= self.epsilonDecayFactor
	
	table.clear(self.advantageHistory)
	
	table.clear(self.gradientHistory)
	
end

function VanillaPolicyGradientModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue
	
end

function VanillaPolicyGradientModel:getLabelFromOutputMatrix(outputMatrix)

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

function VanillaPolicyGradientModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
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

function VanillaPolicyGradientModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function VanillaPolicyGradientModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function VanillaPolicyGradientModel:getCurrentEpsilon()

	return self.currentEpsilon

end

function VanillaPolicyGradientModel:reset()

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	table.clear(self.advantageHistory)
	
	table.clear(self.gradientHistory)

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

function VanillaPolicyGradientModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return VanillaPolicyGradientModel
