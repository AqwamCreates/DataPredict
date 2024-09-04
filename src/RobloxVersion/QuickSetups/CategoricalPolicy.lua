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

CategoricalPolicyQuickSetup = {}

CategoricalPolicyQuickSetup.__index = CategoricalPolicyQuickSetup

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

function CategoricalPolicyQuickSetup.new(numberOfReinforcementsPerEpisode, epsilon, actionSelectionFunction)
	
	local NewCategoricalPolicyQuickSetup = {}
	
	setmetatable(NewCategoricalPolicyQuickSetup, CategoricalPolicyQuickSetup)
	
	NewCategoricalPolicyQuickSetup.isOutputPrinted = true
	
	NewCategoricalPolicyQuickSetup.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewCategoricalPolicyQuickSetup.epsilon = epsilon or defaultEpsilon
	
	NewCategoricalPolicyQuickSetup.currentEpsilon = epsilon or defaultEpsilon
	
	NewCategoricalPolicyQuickSetup.actionSelectionFunction = actionSelectionFunction or defaultActionSelectionFunction
	
	NewCategoricalPolicyQuickSetup.Model = nil
	
	NewCategoricalPolicyQuickSetup.ExperienceReplay = nil
	
	NewCategoricalPolicyQuickSetup.EpsilonValueScheduler = nil
	
	NewCategoricalPolicyQuickSetup.previousFeatureVector = nil
	
	NewCategoricalPolicyQuickSetup.currentNumberOfReinforcements = 0

	NewCategoricalPolicyQuickSetup.currentNumberOfEpisodes = 0
	
	NewCategoricalPolicyQuickSetup.ClassesList = {}
	
	NewCategoricalPolicyQuickSetup.updateFunction = nil
	
	NewCategoricalPolicyQuickSetup.episodeUpdateFunction = nil
	
	return NewCategoricalPolicyQuickSetup
	
end

function CategoricalPolicyQuickSetup:setParameters(numberOfReinforcementsPerEpisode, epsilon, actionSelectionFunction)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon 
	
	self.currentEpsilon = epsilon or self.currentEpsilon
	
	self.actionSelectionFunction = actionSelectionFunction or self.actionSelectionFunction
	
end

function CategoricalPolicyQuickSetup:extendUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function CategoricalPolicyQuickSetup:extendEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function CategoricalPolicyQuickSetup:setPrintOutput(option)

	self.isOutputPrinted = getBooleanOrDefaultOption(option, self.isOutputPrinted)

end

function CategoricalPolicyQuickSetup:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValue(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function CategoricalPolicyQuickSetup:getLabelFromOutputMatrix(outputMatrix)

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

function CategoricalPolicyQuickSetup:selectAction(allOutputsMatrix, ClassesList)
	
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

function CategoricalPolicyQuickSetup:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, childModelNumber)

	if (self.Model == nil) then error("No model!") end
	
	local currentNumberOfReinforcements = self.currentNumberOfReinforcements
	
	local currentNumberOfEpisodes = self.currentNumberOfEpisodes
	
	local ExperienceReplay = self.ExperienceReplay
	
	local EpsilonValueScheduler = self.EpsilonValueScheduler
	
	local currentEpsilon = self.currentEpsilon
	
	local previousFeatureVector = self.previousFeatureVector
	
	local Model = self.Model
	
	local ClassesList = self.ClassesList
	
	local updateFunction = self.updateFunction
	
	local randomProbability = Random.new():NextNumber()
	
	local allOutputsMatrix = Model:predict(currentFeatureVector, true, childModelNumber)

	local action

	local selectedValue

	local temporalDifferenceError

	if (randomProbability < currentEpsilon) then

		local numberOfClasses = #ClassesList

		local randomNumber = Random.new():NextInteger(1, numberOfClasses)

		action = ClassesList[randomNumber]
		
		selectedValue = allOutputsMatrix[1][randomNumber]

	else

		action, selectedValue = self:selectAction(allOutputsMatrix, ClassesList)

	end

	if (previousFeatureVector) then
		
		currentNumberOfReinforcements = currentNumberOfReinforcements + 1

		temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, childModelNumber)
		
		if (updateFunction) then updateFunction(childModelNumber) end

	end

	if (currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then
		
		local episodeUpdateFunction = self.episodeUpdateFunction
		
		currentNumberOfReinforcements = 0
		
		currentNumberOfEpisodes = currentNumberOfEpisodes + 1

		Model:categoricalEpisodeUpdate(childModelNumber)
		
		if episodeUpdateFunction then episodeUpdateFunction(childModelNumber) end

	end

	if (ExperienceReplay) and (previousFeatureVector) then

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			return Model:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end
	
	if (EpsilonValueScheduler) and (previousFeatureVector) then
		
		currentEpsilon = EpsilonValueScheduler:calculate(currentEpsilon)
		
		self.currentEpsilon = currentEpsilon
		
	end
	
	self.currentNumberOfReinforcements = currentNumberOfReinforcements
	
	self.currentNumberOfEpisodes = currentNumberOfEpisodes
	
	self.previousFeatureVector = currentFeatureVector

	if (self.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tEpsilon: " .. currentEpsilon .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, selectedValue

end

function CategoricalPolicyQuickSetup:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function CategoricalPolicyQuickSetup:setModel(Model)

	self.Model = Model

end

function CategoricalPolicyQuickSetup:setEpsilonValueScheduler(EpsilonValueScheduler)

	self.EpsilonValueScheduler = EpsilonValueScheduler

end

function CategoricalPolicyQuickSetup:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function CategoricalPolicyQuickSetup:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function CategoricalPolicyQuickSetup:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function CategoricalPolicyQuickSetup:getCurrentEpsilon()

	return self.currentEpsilon

end

function CategoricalPolicyQuickSetup:getModel()
	
	return self.Model
	
end

function CategoricalPolicyQuickSetup:getExperienceReplay()

	return self.ExperienceReplay

end

function CategoricalPolicyQuickSetup:getEpsilonValueScheduler()

	return self.EpsilonValueScheduler

end


function CategoricalPolicyQuickSetup:getClassesList()
	
	return self.ClassesList
	
end

function CategoricalPolicyQuickSetup:reset()
	
	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon
	
	local Model = self.Model
	
	local ExperienceReplay = self.ExperienceReplay
	
	if (Model) then Model:categoricalReset() end
	
	if (ExperienceReplay) then ExperienceReplay:reset() end
	
end

return CategoricalPolicyQuickSetup