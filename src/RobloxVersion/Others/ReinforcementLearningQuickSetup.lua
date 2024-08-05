--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ReinforcementLearningQuickSetup = {}

ReinforcementLearningQuickSetup.__index = ReinforcementLearningQuickSetup

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0

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

local function calculateProbability(outputMatrix)

	local meanVector = AqwamMatrixLibrary:horizontalMean(outputMatrix)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(outputMatrix)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(outputMatrix, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local zScoreSquaredVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, zScoreSquaredVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:applyFunction(math.exp, probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

function ReinforcementLearningQuickSetup.new(numberOfReinforcementsPerEpisode, epsilon, actionSelectionFunction)
	
	local NewReinforcementLearningQuickSetup = {}
	
	setmetatable(NewReinforcementLearningQuickSetup, ReinforcementLearningQuickSetup)
	
	NewReinforcementLearningQuickSetup.isOutputPrinted = true
	
	NewReinforcementLearningQuickSetup.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewReinforcementLearningQuickSetup.epsilon = epsilon or defaultEpsilon
	
	NewReinforcementLearningQuickSetup.currentEpsilon = epsilon or defaultEpsilon
	
	NewReinforcementLearningQuickSetup.actionSelectionFunction = actionSelectionFunction or defaultActionSelectionFunction
	
	NewReinforcementLearningQuickSetup.Model = nil
	
	NewReinforcementLearningQuickSetup.ExperienceReplay = nil
	
	NewReinforcementLearningQuickSetup.EpsilonValueScheduler = nil
	
	NewReinforcementLearningQuickSetup.previousFeatureVector = nil
	
	NewReinforcementLearningQuickSetup.currentNumberOfReinforcements = 0

	NewReinforcementLearningQuickSetup.currentNumberOfEpisodes = 0
	
	NewReinforcementLearningQuickSetup.ClassesList = {}
	
	NewReinforcementLearningQuickSetup.updateFunction = nil
	
	NewReinforcementLearningQuickSetup.episodeUpdateFunction = nil
	
	return NewReinforcementLearningQuickSetup
	
end

function ReinforcementLearningQuickSetup:setParameters(numberOfReinforcementsPerEpisode, epsilon, actionSelectionFunction)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon 
	
	self.currentEpsilon = epsilon or self.currentEpsilon
	
	self.actionSelectionFunction = actionSelectionFunction or self.actionSelectionFunction
	
end

function ReinforcementLearningQuickSetup:extendUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningQuickSetup:extendEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function ReinforcementLearningQuickSetup:setPrintOutput(option)

	self.isOutputPrinted = getBooleanOrDefaultOption(option, self.isOutputPrinted)

end

function ReinforcementLearningQuickSetup:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValue(outputVector)

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

function ReinforcementLearningQuickSetup:selectAction(allOutputsMatrix, ClassesList)
	
	local action
	
	local selectedValue
	
	local actionSelectionFunction = self.actionSelectionFunction
	
	if (actionSelectionFunction == "Maximum") then
		
		local actionVector, selectedValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)
		
		action = actionVector[1][1]

		selectedValue = selectedValueVector[1][1]
		
	elseif (actionSelectionFunction == "Sample") then
		
		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local actionIndex = sampleAction(actionProbabilityVector)
		
		action = ClassesList[actionIndex]
		
		selectedValue = allOutputsMatrix[1][actionIndex]
		
	else
		
		error("Invalid action selection function!")
		
	end
	
	return action, selectedValue
	
end

function ReinforcementLearningQuickSetup:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, childModelNumber)

	if (self.Model == nil) then error("No model!") end
	
	local randomProbability = Random.new():NextNumber()
	
	local ExperienceReplay = self.ExperienceReplay
	
	local EpsilonValueScheduler = self.EpsilonValueScheduler
	
	local currentEpsilon = self.currentEpsilon
	
	local previousFeatureVector = self.previousFeatureVector
	
	local Model = self.Model
	
	local ClassesList = self.ClassesList
	
	local updateFunction = self.updateFunction

	local action

	local selectedValue

	local allOutputsMatrix

	local temporalDifferenceError

	self.currentNumberOfReinforcements = self.currentNumberOfReinforcements + 1

	if (randomProbability < currentEpsilon) then

		local numberOfClasses = #ClassesList

		local randomNumber = Random.new():NextInteger(1, numberOfClasses)

		action = ClassesList[randomNumber]

		allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)

		allOutputsMatrix[1][randomNumber] = randomProbability

	else
		
		allOutputsMatrix = Model:predict(currentFeatureVector, true, childModelNumber)

		action, selectedValue = self:selectAction(allOutputsMatrix, ClassesList)

	end

	if (previousFeatureVector) then 

		temporalDifferenceError = Model:update(previousFeatureVector, action, rewardValue, currentFeatureVector, childModelNumber) 

	end

	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then
		
		local episodeUpdateFunction = self.episodeUpdateFunction
		
		self.currentNumberOfReinforcements = 0

		Model:episodeUpdate(childModelNumber)
		
		if episodeUpdateFunction then episodeUpdateFunction(childModelNumber) end

	end

	if (ExperienceReplay) and (previousFeatureVector) then

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			return Model:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end
	
	if updateFunction then updateFunction(childModelNumber) end
	
	if (EpsilonValueScheduler) then
		
		currentEpsilon = EpsilonValueScheduler:calculate(currentEpsilon)
		
		self.currentEpsilon = currentEpsilon
		
	end

	self.previousFeatureVector = currentFeatureVector

	if (self.isOutputPrinted) then print("Episode: " .. self.currentNumberOfEpisodes .. "\t\tEpsilon: " .. currentEpsilon .. "\t\tReinforcement Count: " .. self.currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, selectedValue

end

function ReinforcementLearningQuickSetup:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function ReinforcementLearningQuickSetup:setModel(Model)

	self.Model = Model

end

function ReinforcementLearningQuickSetup:setEpsilonValueScheduler(EpsilonValueScheduler)

	self.EpsilonValueScheduler = EpsilonValueScheduler

end

function ReinforcementLearningQuickSetup:setClassesList(ClassesList)

	self.ClassesList = ClassesList

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

function ReinforcementLearningQuickSetup:getModel()
	
	return self.Model
	
end

function ReinforcementLearningQuickSetup:getExperienceReplay()

	return self.ExperienceReplay

end

function ReinforcementLearningQuickSetup:getEpsilonValueScheduler()

	return self.EpsilonValueScheduler

end


function ReinforcementLearningQuickSetup:getClassesList()
	
	return self.ClassesList
	
end

function ReinforcementLearningQuickSetup:reset()
	
	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	local ExperienceReplay = self.ExperienceReplay
	
	if (ExperienceReplay) then ExperienceReplay:reset() end
	
end

return ReinforcementLearningQuickSetup