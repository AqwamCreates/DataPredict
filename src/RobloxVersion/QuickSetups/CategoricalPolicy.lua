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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ReinforcementLearningBaseQuickSetup = require(script.Parent.ReinforcementLearningBaseQuickSetup)

CategoricalPolicyQuickSetup = {}

CategoricalPolicyQuickSetup.__index = CategoricalPolicyQuickSetup

setmetatable(CategoricalPolicyQuickSetup, ReinforcementLearningBaseQuickSetup)

local defaultActionSelectionFunction = "Maximum"

local function selectIndexWithHighestValue(valueVector)
	
	local selectedIndex = 1
	
	local highestValue = -math.huge
	
	for index, value in ipairs(valueVector[1]) do

		if (highestValue > value) then

			highestValue = value

			selectedIndex = index

		end

	end
	
	return selectedIndex
	
end

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

local function sample(valueVector)
	
	local probabilityVector = calculateProbability(valueVector)

	local totalProbability = 0

	for _, probability in ipairs(probabilityVector[1]) do

		totalProbability = totalProbability + probability

	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0

	local selectedIndex = 1

	for i, probability in ipairs(probabilityVector[1]) do

		cumulativeProbability = cumulativeProbability + probability

		if (randomValue > cumulativeProbability) then continue end

		selectedIndex = i

		break

	end

	return selectedIndex

end

local selectActionFunctionList = {

	["Maximum"] = selectIndexWithHighestValue,

	["Sample"] = sample

}

function CategoricalPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewCategoricalPolicyQuickSetup = ReinforcementLearningBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewCategoricalPolicyQuickSetup, CategoricalPolicyQuickSetup)
	
	NewCategoricalPolicyQuickSetup:setName("CategoricalPolicyQuickSetup")
	
	NewCategoricalPolicyQuickSetup.actionSelectionFunction = parameterDictionary.actionSelectionFunction or defaultActionSelectionFunction
	
	NewCategoricalPolicyQuickSetup.previousAction = parameterDictionary.previousAction
	
	NewCategoricalPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue, returnOriginalOutput)
		
		local Model = NewCategoricalPolicyQuickSetup.Model
		
		if (not Model) then error("No model!") end
		
		local numberOfReinforcementsPerEpisode = NewCategoricalPolicyQuickSetup.numberOfReinforcementsPerEpisode

		local currentNumberOfReinforcements = NewCategoricalPolicyQuickSetup.currentNumberOfReinforcements

		local currentNumberOfEpisodes = NewCategoricalPolicyQuickSetup.currentNumberOfEpisodes

		local EpsilonValueScheduler = NewCategoricalPolicyQuickSetup.EpsilonValueScheduler

		local currentEpsilon = NewCategoricalPolicyQuickSetup.currentEpsilon
		
		local ExperienceReplay = NewCategoricalPolicyQuickSetup.ExperienceReplay
		
		local previousFeatureVector = NewCategoricalPolicyQuickSetup.previousFeatureVector
		
		local previousAction =  NewCategoricalPolicyQuickSetup.previousAction

		local ClassesList = Model:getClassesList()

		local randomProbability = Random.new():NextNumber()

		local actionVector = Model:predict(currentFeatureVector, true)
		
		local terminalStateValue = 0

		local actionIndex

		local action

		local actionValue

		local temporalDifferenceError

		if (randomProbability < currentEpsilon) then

			actionIndex = Random.new():NextInteger(1, #ClassesList)

		else

			actionIndex = selectActionFunctionList[NewCategoricalPolicyQuickSetup.actionSelectionFunction](actionVector)

		end

		action = ClassesList[actionIndex]

		actionValue = actionVector[1][actionIndex]
		
		if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then terminalStateValue = 1 end

		if (previousFeatureVector) then
			
			local updateFunction = NewCategoricalPolicyQuickSetup.updateFunction

			currentNumberOfReinforcements = currentNumberOfReinforcements + 1

			temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction() end

		end

		if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then

			local episodeUpdateFunction = NewCategoricalPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction() end

		end

		if (ExperienceReplay) and (previousFeatureVector) then

			ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

				return Model:categoricalUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			end)

		end

		if (EpsilonValueScheduler) and (previousFeatureVector) then

			currentEpsilon = EpsilonValueScheduler:calculate(currentEpsilon)

			NewCategoricalPolicyQuickSetup.currentEpsilon = currentEpsilon

		end

		NewCategoricalPolicyQuickSetup.currentNumberOfReinforcements = currentNumberOfReinforcements

		NewCategoricalPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes

		NewCategoricalPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewCategoricalPolicyQuickSetup.previousAction = action

		if (NewCategoricalPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tEpsilon: " .. currentEpsilon .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	return NewCategoricalPolicyQuickSetup
	
end

return CategoricalPolicyQuickSetup