local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ReinforcementLearningQuickSetup = {}

ReinforcementLearningQuickSetup.__index = ReinforcementLearningQuickSetup

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultActionSelectionFunction = "Maximum"

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

function ReinforcementLearningQuickSetup.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, actionSelectionFunction)
	
	local NewReinforcementLearningQuickSetup = {}
	
	setmetatable(NewReinforcementLearningQuickSetup, ReinforcementLearningQuickSetup)
	
	NewReinforcementLearningQuickSetup.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewReinforcementLearningQuickSetup.epsilon = epsilon or defaultEpsilon

	NewReinforcementLearningQuickSetup.epsilonDecayFactor = epsilonDecayFactor or defaultEpsilon
	
	NewReinforcementLearningQuickSetup.currentEpsilon = epsilon or defaultEpsilon
	
	NewReinforcementLearningQuickSetup.actionSelectionFunction = actionSelectionFunction or defaultActionSelectionFunction
	
	NewReinforcementLearningQuickSetup.Model = nil
	
	NewReinforcementLearningQuickSetup.ExperienceReplay = nil
	
	NewReinforcementLearningQuickSetup.previousFeatureVector = nil
	
	NewReinforcementLearningQuickSetup.currentNumberOfReinforcements = 0

	NewReinforcementLearningQuickSetup.currentNumberOfEpisodes = 0
	
	NewReinforcementLearningQuickSetup.ClassesList = {}
	
	return NewReinforcementLearningQuickSetup
	
end

function ReinforcementLearningQuickSetup:setParameters(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, actionSelectionFunction)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon 

	self.epsilonDecayFactor = epsilonDecayFactor or self.epsilonDecayFactor
	
	self.currentEpsilon = epsilon or self.currentEpsilon
	
	self.actionSelectionFunction = actionSelectionFunction or self.actionSelectionFunction
	
end

function ReinforcementLearningQuickSetup:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function ReinforcementLearningQuickSetup:setModel(Model)

	self.Model = Model or self.Model

end

function ReinforcementLearningQuickSetup:setClassesList(classesList)

	self.ClassesList = classesList

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function ReinforcementLearningQuickSetup:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ReinforcementLearningQuickSetup:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function ReinforcementLearningQuickSetup:getLabelFromOutputMatrix(outputMatrix)

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

function ReinforcementLearningQuickSetup:selectAction(currentFeatureVector, classesList)
	
	local allOutputsMatrix = self.Model:predict(currentFeatureVector, true)
	
	local action
	
	local selectedValue
	
	if (self.actionSelectionFunction == "Maximum") then
		
		local actionVector, selectedValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)
		
		action = actionVector[1][1]

		selectedValue = selectedValueVector[1][1]
		
	elseif (self.actionSelectionFunction == "Sample") then
		
		local actionIndex = sampleAction(allOutputsMatrix)
		
		action = classesList[actionIndex]
		
		selectedValue = allOutputsMatrix[1][actionIndex]
		
	end
	
	return action, selectedValue
	
end

function ReinforcementLearningQuickSetup:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)

	if (self.Model == nil) then error("No model!") end
	
	local ExperienceReplay = self.ExperienceReplay
	
	local previousFeatureVector = self.previousFeatureVector
	
	local Model = self.Model
	
	local ClassesList = self.ClassesList

	self.currentNumberOfReinforcements += 1

	local action

	local selectedValue

	local allOutputsMatrix

	local temporalDifferenceError

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < self.currentEpsilon) then

		local numberOfClasses = #ClassesList

		local randomNumber = Random.new():NextInteger(1, numberOfClasses)

		action = ClassesList[randomNumber]

		allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		action, selectedValue = self:selectAction(currentFeatureVector, ClassesList)

	end

	if (previousFeatureVector) then 

		temporalDifferenceError = Model:update(previousFeatureVector, action, rewardValue, currentFeatureVector) 

	end

	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then

		Model:episodeUpdate()

	end

	if (ExperienceReplay) and (previousFeatureVector) then

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			return Model:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput) then print("Episode: " .. self.currentNumberOfEpisodes .. "\t\tEpsilon: " .. self.currentEpsilon .. "\t\tReinforcement Count: " .. self.currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, selectedValue

end

function ReinforcementLearningQuickSetup:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function ReinforcementLearningQuickSetup:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function ReinforcementLearningQuickSetup:getCurrentEpsilon()

	return self.currentEpsilon

end

function ReinforcementLearningQuickSetup:reset()
	
	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end
	
end

return ReinforcementLearningQuickSetup
