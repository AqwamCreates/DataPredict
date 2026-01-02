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

local defaultShareCurrentTemperature = true

local defaultShareCurrentCValue = true

local defaultShareEpsilonValueScheduler = true

local defaultShareTemperatureValueScheduler = true

local defaultShareCValueValueScheduler = true

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

	NewParallelCategoricalPolicyQuickSetup.shareCurrentTemperature = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentTemperature or defaultShareCurrentTemperature)

	NewParallelCategoricalPolicyQuickSetup.shareCurrentCValue = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentCValue or defaultShareCurrentCValue)

	NewParallelCategoricalPolicyQuickSetup.shareEpsilonValueScheduler = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareEpsilonValueScheduler or defaultShareEpsilonValueScheduler)

	NewParallelCategoricalPolicyQuickSetup.shareTemperatureValueScheduler = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareTemperatureValueScheduler or defaultShareTemperatureValueScheduler)

	NewParallelCategoricalPolicyQuickSetup.shareCValueValueScheduler = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCValueValueScheduler or defaultShareCValueValueScheduler)

	NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)

	NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewParallelCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)
	
	-- Dictionaries

	NewParallelCategoricalPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.EligibilityTraceDictionary = parameterDictionary.EligibilityTraceDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.previousActionDictionary = parameterDictionary.previousActionDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = parameterDictionary.selectedActionCountVectorDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary = parameterDictionary.currentEpsilonDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.currentTemperatureDictionary = parameterDictionary.currentTemperatureDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.currentCValueDictionary = parameterDictionary.currentCValueDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary = parameterDictionary.EpsilonValueSchedulerDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.TemperatureValueSchedulerDictionary = parameterDictionary.TemperatureValueSchedulerDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.CValueValueSchedulerDictionary = parameterDictionary.CValueValueSchedulerDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}

	NewParallelCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	NewParallelCategoricalPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue, returnOriginalOutput)
		
		local Model = NewParallelCategoricalPolicyQuickSetup.Model

		if (not Model) then error("No model.") end
		
		local isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")
		
		local selectedActionCountVectorIndex = (NewParallelCategoricalPolicyQuickSetup.shareSelectedActionCountVector and 1) or agentIndex

		local currentEpsilonIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentEpsilon and 1) or agentIndex

		local currentTemperatureIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentTemperature and 1) or agentIndex

		local currentCValueIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentCValue and 1) or agentIndex

		local epsilonValueSchedulerIndex = (NewParallelCategoricalPolicyQuickSetup.shareEpsilonValueScheduler and 1) or agentIndex

		local temperatureValueSchedulerIndex = (NewParallelCategoricalPolicyQuickSetup.shareTemperatureValueScheduler and 1) or agentIndex

		local cValueValueSchedulerIndex = (NewParallelCategoricalPolicyQuickSetup.shareCValueValueScheduler and 1) or agentIndex

		local experienceReplayIndex = (NewParallelCategoricalPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex

		local eligibilityTraceIndex = (NewParallelCategoricalPolicyQuickSetup.shareEligibilityTrace and 1) or agentIndex
		
		local numberOfReinforcementsIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex

		local numberOfEpisodesIndex = (NewParallelCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex
		
		local previousFeatureVectorDictionary = NewParallelCategoricalPolicyQuickSetup.previousFeatureVectorDictionary
		
		local previousActionDictionary = NewParallelCategoricalPolicyQuickSetup.previousActionDictionary
		
		local selectedActionCountVectorDictionary = NewParallelCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary
		
		local currentEpsilonDictionary = NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary

		local currentTemperatureDictionary = NewParallelCategoricalPolicyQuickSetup.currentTemperatureDictionary

		local currentCValueDictionary = NewParallelCategoricalPolicyQuickSetup.currentCValueDictionary
		
		local currentNumberOfReinforcementsDictionary = NewParallelCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary
		
		local numberOfReinforcementsPerEpisode = NewParallelCategoricalPolicyQuickSetup.numberOfReinforcementsPerEpisode
		
		local currentNumberOfEpisodesDictionary = NewParallelCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary
		
		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]
		
		local previousAction = previousActionDictionary[agentIndex]
		
		local selectedActionCountVector = selectedActionCountVectorDictionary[selectedActionCountVectorIndex]
		
		local currentEpsilon = currentEpsilonDictionary[currentEpsilonIndex]
		
		local currentTemperature = currentTemperatureDictionary[currentTemperatureIndex]
		
		local currentCValue = currentCValueDictionary[currentCValueIndex]
		
		local EpsilonValueScheduler = NewParallelCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary[epsilonValueSchedulerIndex]
		
		local TemperatureValueScheduler = NewParallelCategoricalPolicyQuickSetup.TemperatureValueSchedulerDictionary[temperatureValueSchedulerIndex]
		
		local CValueValueScheduler = NewParallelCategoricalPolicyQuickSetup.CValueValueSchedulerDictionary[cValueValueSchedulerIndex]
		
		local ExperienceReplay = NewParallelCategoricalPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]
		
		local EligibilityTrace = NewParallelCategoricalPolicyQuickSetup.EligibilityTraceDictionary[eligibilityTraceIndex]
		
		local currentNumberOfReinforcements = (currentNumberOfReinforcementsDictionary[numberOfReinforcementsIndex] or 0) + 1
		
		local currentNumberOfEpisodes = currentNumberOfEpisodesDictionary[numberOfEpisodesIndex] or 1
		
		local ActionsList = Model:getActionsList()

		local currentActionVector = Model:predict(currentFeatureVector, true)
		
		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)

		local terminalStateValue = (isEpisodeEnd and 1) or 0

		if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

		local currentActionIndex, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue = NewParallelCategoricalPolicyQuickSetup:selectAction(currentActionVector, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue, EpsilonValueScheduler, TemperatureValueScheduler, CValueValueScheduler, currentNumberOfReinforcements)

		local currentAction = ActionsList[currentActionIndex]

		local currentActionValue = currentActionVector[1][currentActionIndex]
		
		local temporalDifferenceError

		if (previousFeatureVector) then

			local updateFunction = NewParallelCategoricalPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			local episodeUpdateFunction = NewParallelCategoricalPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if (episodeUpdateFunction) then episodeUpdateFunction(terminalStateValue) end

		end

		if (ExperienceReplay) and (previousFeatureVector) then

			ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

				return Model:categoricalUpdate(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

			end)

		end
		
		previousActionDictionary[agentIndex] = currentAction

		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector

		selectedActionCountVectorDictionary[selectedActionCountVectorIndex] = selectedActionCountVector

		currentEpsilonDictionary[currentEpsilonIndex] = currentEpsilon

		currentTemperatureDictionary[currentTemperature] = currentTemperature

		currentCValueDictionary[currentCValueIndex] = currentCValue

		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes
		
		if (NewParallelCategoricalPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		if (returnOriginalOutput) then return currentActionVector end

		return currentAction, currentActionValue
		
	end)
	
	NewParallelCategoricalPolicyQuickSetup:setResetFunction(function()

		NewParallelCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.previousActionDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.currentTemperatureictionary = {}

		NewParallelCategoricalPolicyQuickSetup.currentEpsilonDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary = {}

		NewParallelCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary = {}

		for _, EpsilonValueScheduler in ipairs(NewParallelCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary) do EpsilonValueScheduler:reset() end

		for _, TemperatureValueScheduler in ipairs(NewParallelCategoricalPolicyQuickSetup.TemperatureValueSchedulerDictionary) do TemperatureValueScheduler:reset() end

		for _, CValueValueScheduler in ipairs(NewParallelCategoricalPolicyQuickSetup.CValueValueSchedulerDictionary) do CValueValueScheduler:reset() end

		for _, ExperienceReplay in ipairs(NewParallelCategoricalPolicyQuickSetup.ExperienceReplayDictionary) do ExperienceReplay:reset() end

		for _, EligibilityTrace in ipairs(NewParallelCategoricalPolicyQuickSetup.EligibilityTraceDictionary) do EligibilityTrace:reset() end

	end)
	
	return NewParallelCategoricalPolicyQuickSetup
	
end

return ParallelCategoricalPolicyQuickSetup
