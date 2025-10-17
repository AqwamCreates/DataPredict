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

ParallelCategoricalPolicyQuickSetup = {}

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

		local terminalStateValue = 0

		local temporalDifferenceError

		if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

		local actionIndex, selectedActionCountVector = NewParallelCategoricalPolicyQuickSetup:selectAction(actionVector, selectedActionCountVector, currentEpsilon, EpsilonValueScheduler, currentNumberOfReinforcements)

		local action = ActionsList[actionIndex]

		local actionValue = actionVector[1][actionIndex]

		if (isEpisodeEnd) then terminalStateValue = 1 end

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

			ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

				return Model:categoricalUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

			end)

		end

		previousActionDictionary[agentIndex] = action
		
		currentEpsilonDictionary[actionIndex] = currentEpsilon

		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes

		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector
		
		selectedActionCountVectorDictionary[selectedActionCountVectorIndex] = selectedActionCountVector
		
		if (NewParallelCategoricalPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	return NewParallelCategoricalPolicyQuickSetup
	
end

function ParallelCategoricalPolicyQuickSetup:reset()
	
	self.previousFeatureVectorDictionary = {}

	self.previousActionDictionary = {}
	
	self.selectedActionCountVectorDictionary = {}

	self.currentNumberOfReinforcementsDictionary  = {}

	self.currentNumberOfEpisodesDictionary  = {}
	
	for _, ExperienceReplay in ipairs(self.ExperienceReplayDictionary) do ExperienceReplay:reset() end
	
	for _, EligibilityTrace in ipairs(self.EligibilityTraceDictionary) do EligibilityTrace:reset() end
		
end

return ParallelCategoricalPolicyQuickSetup
