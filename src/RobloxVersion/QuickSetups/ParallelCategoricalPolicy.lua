--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local CategoricalPolicyBaseQuickSetup = require(script.Parent.CategoricalPolicyBaseQuickSetup)

local ParallelCategoricalPolicyQuickSetup = {}

ParallelCategoricalPolicyQuickSetup.__index = ParallelCategoricalPolicyQuickSetup

setmetatable(ParallelCategoricalPolicyQuickSetup, CategoricalPolicyBaseQuickSetup)

local defaultShareExperienceReplay = false

local defaultShareEligibilityTrace = false

local defaultShareSelectedActionCountVector = false

local defaultShareCurrentEpsilon = true

local defaultShareEpsilonValueScheduler = true

local defaultShareCurrentNumberOfReinforcements = false

local defaultShareCurrentNumberOfEpisodes = false

function ParallelCategoricalPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewParallelCategoricalPolicyQuickSetup = CategoricalPolicyBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewParallelCategoricalPolicyQuickSetup, ParallelCategoricalPolicyQuickSetup)
	
	NewParallelCategoricalPolicyQuickSetup:setName("ParallelCategoricalPolicyQuickSetup")
	
	-- Share toggles
	
	NewParallelCategoricalPolicyQuickSetup.shareExperienceReplay = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)
	
	NewParallelCategoricalPolicyQuickSetup.shareEligibilityTrace = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareEligibilityTrace or defaultShareEligibilityTrace)
	
	NewParallelCategoricalPolicyQuickSetup.shareSelectedActionCountVector = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareSelectedActionCountVector or defaultShareSelectedActionCountVector)
	
	NewParallelCategoricalPolicyQuickSetup.shareCurrentEpsilon = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentEpsilon or defaultShareCurrentEpsilon)
	
	NewParallelCategoricalPolicyQuickSetup.shareEpsilonValueScheduler = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentEpsilon or defaultShareEpsilonValueScheduler)
	
	NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)
	
	NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)
	
	-- Dictionaries

	NewParallelCategoricalPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.EligibilityTraceDictionary = parameterDictionary.EligibilityTraceDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.previousActionDictionary = parameterDictionary.previousActionDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = parameterDictionary.selectedActionCountVectorDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary = parameterDictionary.currentEpsilonDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary = parameterDictionary.EpsilonValueSchedulerDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue, returnOriginalOutput)
		
		local Model = NewParallelCategoricalPolicyQuickSetup.Model

		if (not Model) then error("No model.") end
		
		local isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")
		
		local numberOfReinforcementsPerEpisode = NewParallelCategoricalPolicyQuickSetup.numberOfReinforcementsPerEpisode
		
		local experienceReplayIndex = (NewParallelCategoricalPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex
		
		local eligibilityTraceIndex = (NewParallelCategoricalPolicyQuickSetup.shareEligibilityTrace and 1) or agentIndex
		
		local selectedActionCountVectorIndex = (NewParallelCategoricalPolicyQuickSetup.shareSelectedActionCountVector and 1) or agentIndex
		
		local currentEpsilonIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentEpsilon and 1) or agentIndex
		
		local epsilonValueSchedulerIndex = (NewParallelCategoricalPolicyQuickSetup.shareEpsilonValueScheduler and 1) or agentIndex
		
		local currentEpsilonSchedulerIndex = (NewParallelCategoricalPolicyQuickSetup.shareEpsilonValueScheduler and 1) or agentIndex
		
		local numberOfReinforcementsIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex
		
		local numberOfEpisodesIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex
		
		local previousFeatureVectorDictionary = NewParallelCategoricalPolicyQuickSetup.previousFeatureVectorDictionary
		
		local previousActionDictionary = NewParallelCategoricalPolicyQuickSetup.previousActionDictionary
		
		local selectedActionCountVectorDictionary = NewParallelCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary
		
		local currentEpsilonDictionary = NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary
		
		local currentNumberOfReinforcementsDictionary = NewParallelCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary
		
		local currentNumberOfEpisodesDictionary = NewParallelCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary
		
		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]
		
		local previousAction = previousActionDictionary[agentIndex]
		
		local selectedActionCountVector = selectedActionCountVectorDictionary[selectedActionCountVectorIndex]
		
		local currentEpsilon = currentEpsilonDictionary[currentEpsilonIndex]
		
		local EpsilonValueScheduler = NewParallelCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary[epsilonValueSchedulerIndex]
		
		local ExperienceReplay = NewParallelCategoricalPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]
		
		local EligibilityTrace = NewParallelCategoricalPolicyQuickSetup.EligibilityTraceDictionary[eligibilityTraceIndex]
		
		local currentNumberOfReinforcements = (currentNumberOfReinforcementsDictionary[numberOfReinforcementsIndex] or 0) + 1
		
		local currentNumberOfEpisodes = currentNumberOfEpisodesDictionary[numberOfEpisodesIndex] or 1
		
		local ActionsList = Model:getActionsList()

		local actionVector = Model:predict(currentFeatureVector, true)
		
		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)

		local terminalStateValue = (isEpisodeEnd and 1) or 0

		if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

		local actionIndex, selectedActionCountVector = NewParallelCategoricalPolicyQuickSetup:selectAction(actionVector, selectedActionCountVector, currentEpsilon, EpsilonValueScheduler, currentNumberOfReinforcements)

		local currentAction = ActionsList[actionIndex]

		local currentActionValue = actionVector[1][actionIndex]
		
		local temporalDifferenceError

		if (previousFeatureVector) then

			local updateFunction = NewParallelCategoricalPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			local episodeUpdateFunction = NewParallelCategoricalPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue) end

		end

		if (ExperienceReplay) and (previousFeatureVector) then

			ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

				return Model:categoricalUpdate(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

			end)

		end
		
		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector

		previousActionDictionary[agentIndex] = currentAction
		
		selectedActionCountVectorDictionary[selectedActionCountVectorIndex] = selectedActionCountVector
		
		currentEpsilonDictionary[actionIndex] = currentEpsilon

		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes
		
		if (NewParallelCategoricalPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		if (returnOriginalOutput) then return actionVector end

		return currentAction, currentActionValue
		
	end)
	
	NewParallelCategoricalPolicyQuickSetup:setResetFunction(function()

		NewParallelCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.previousActionDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = {}
		
		NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary = {}
		
		for _, EpsilonValueScheduler in ipairs(NewParallelCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary) do EpsilonValueScheduler:reset() end

		NewParallelCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary  = {}

		NewParallelCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary  = {}

		for _, ExperienceReplay in ipairs(NewParallelCategoricalPolicyQuickSetup.ExperienceReplayDictionary) do ExperienceReplay:reset() end

		for _, EligibilityTrace in ipairs(NewParallelCategoricalPolicyQuickSetup.EligibilityTraceDictionary) do EligibilityTrace:reset() end

	end)
	
	return NewParallelCategoricalPolicyQuickSetup
	
end

return ParallelCategoricalPolicyQuickSetup
