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

local CategoricalPolicyBaseQuickSetup = require("QuickSetup_CategoricalPolicyBaseQuickSetup")

local QueuedCategoricalPolicyQuickSetup = {}

QueuedCategoricalPolicyQuickSetup.__index = QueuedCategoricalPolicyQuickSetup

setmetatable(QueuedCategoricalPolicyQuickSetup, CategoricalPolicyBaseQuickSetup)

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

function QueuedCategoricalPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewQueuedCategoricalPolicyQuickSetup = CategoricalPolicyBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewQueuedCategoricalPolicyQuickSetup, QueuedCategoricalPolicyQuickSetup)
	
	NewQueuedCategoricalPolicyQuickSetup:setName("QueuedCategoricalPolicyQuickSetup")
	
	-- Share toggles

	NewQueuedCategoricalPolicyQuickSetup.shareExperienceReplay = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)

	NewQueuedCategoricalPolicyQuickSetup.shareEligibilityTrace = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareEligibilityTrace or defaultShareEligibilityTrace)

	NewQueuedCategoricalPolicyQuickSetup.shareSelectedActionCountVector = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareSelectedActionCountVector or defaultShareSelectedActionCountVector)

	NewQueuedCategoricalPolicyQuickSetup.shareCurrentEpsilon = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentEpsilon or defaultShareCurrentEpsilon)
	
	NewQueuedCategoricalPolicyQuickSetup.shareCurrentTemperature = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentTemperature or defaultShareCurrentTemperature)
	
	NewQueuedCategoricalPolicyQuickSetup.shareCurrentCValue = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentCValue or defaultShareCurrentCValue)

	NewQueuedCategoricalPolicyQuickSetup.shareEpsilonValueScheduler = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareEpsilonValueScheduler or defaultShareEpsilonValueScheduler)
	
	NewQueuedCategoricalPolicyQuickSetup.shareTemperatureValueScheduler = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareTemperatureValueScheduler or defaultShareTemperatureValueScheduler)
	
	NewQueuedCategoricalPolicyQuickSetup.shareCValueValueScheduler = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCValueValueScheduler or defaultShareCValueValueScheduler)

	NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)

	NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)

	-- Dictionaries

	NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary = parameterDictionary.EligibilityTraceDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary = parameterDictionary.previousActionDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = parameterDictionary.selectedActionCountVectorDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.currentEpsilonDictionary = parameterDictionary.currentEpsilonDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.currentTemperatureDictionary = parameterDictionary.currentTemperatureDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.currentCValueDictionary = parameterDictionary.currentCValueDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary = parameterDictionary.EpsilonValueSchedulerDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.TemperatureValueSchedulerDictionary = parameterDictionary.TemperatureValueSchedulerDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.CValueValueSchedulerDictionary = parameterDictionary.CValueValueSchedulerDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	-- Queues
	
	NewQueuedCategoricalPolicyQuickSetup.inputQueueArray = parameterDictionary.inputQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.agentIndexOutputQueueArray = parameterDictionary.agentIndexOutputQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.outputQueueArray = parameterDictionary.outputQueueArray or {}
	
	-- Debounce
	
	NewQueuedCategoricalPolicyQuickSetup.isRunning = false
	
	NewQueuedCategoricalPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue, returnOriginalOutput)
		
		if (not NewQueuedCategoricalPolicyQuickSetup.isRunning) then error("Not currently running.") end

		local selectedActionCountVectorIndex = (NewQueuedCategoricalPolicyQuickSetup.shareSelectedActionCountVector and 1) or agentIndex

		local currentEpsilonIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentEpsilon and 1) or agentIndex
		
		local currentTemperatureIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentTemperature and 1) or agentIndex
		
		local currentCValueIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentCValue and 1) or agentIndex

		local epsilonValueSchedulerIndex = (NewQueuedCategoricalPolicyQuickSetup.shareEpsilonValueScheduler and 1) or agentIndex
		
		local temperatureValueSchedulerIndex = (NewQueuedCategoricalPolicyQuickSetup.shareTemperatureValueScheduler and 1) or agentIndex
		
		local cValueValueSchedulerIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCValueValueScheduler and 1) or agentIndex
		
		local experienceReplayIndex = (NewQueuedCategoricalPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex

		local eligibilityTraceIndex = (NewQueuedCategoricalPolicyQuickSetup.shareEligibilityTrace and 1) or agentIndex

		local numberOfReinforcementsIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex

		local numberOfEpisodesIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex

		local previousFeatureVectorDictionary = NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary

		local previousActionDictionary = NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary

		local selectedActionCountVectorDictionary = NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary

		local currentEpsilonDictionary = NewQueuedCategoricalPolicyQuickSetup.currentEpsilonDictionary
		
		local currentTemperatureDictionary = NewQueuedCategoricalPolicyQuickSetup.currentTemperatureDictionary
		
		local currentCValueDictionary = NewQueuedCategoricalPolicyQuickSetup.currentCValueDictionary

		local currentNumberOfReinforcementsDictionary = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary

		local currentNumberOfEpisodesDictionary = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary

		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]

		local previousAction = previousActionDictionary[agentIndex]

		local selectedActionCountVector = selectedActionCountVectorDictionary[selectedActionCountVectorIndex]

		local currentEpsilon = currentEpsilonDictionary[currentEpsilonIndex]
		
		local currentTemperature = currentTemperatureDictionary[currentTemperatureIndex]
		
		local currentCValue = currentCValueDictionary[currentCValueIndex]

		local EpsilonValueScheduler = NewQueuedCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary[epsilonValueSchedulerIndex]
		
		local TemperatureValueScheduler = NewQueuedCategoricalPolicyQuickSetup.TemperatureValueSchedulerDictionary[temperatureValueSchedulerIndex]
		
		local CValueValueScheduler = NewQueuedCategoricalPolicyQuickSetup.CValueValueSchedulerDictionary[temperatureValueSchedulerIndex]

		local ExperienceReplay = NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]

		local EligibilityTrace = NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary[eligibilityTraceIndex]

		local currentNumberOfReinforcements = (currentNumberOfReinforcementsDictionary[numberOfReinforcementsIndex] or 0) + 1

		local currentNumberOfEpisodes = currentNumberOfEpisodesDictionary[numberOfEpisodesIndex] or 1
		
		local terminalStateValue 
		
		local isEpisodeEnd
		
		if (currentNumberOfReinforcements >= NewQueuedCategoricalPolicyQuickSetup.numberOfReinforcementsPerEpisode) then
			
			isEpisodeEnd = true
			
			terminalStateValue = 1

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			currentNumberOfReinforcements = 0

		else
			
			isEpisodeEnd = false
			
			terminalStateValue = 0

			currentNumberOfReinforcements = currentNumberOfReinforcements + 1

		end
		
		local inputArray = {agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, ExperienceReplay, EligibilityTrace, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue, EpsilonValueScheduler, TemperatureValueScheduler, CValueValueScheduler, currentNumberOfReinforcements}
		
		table.insert(NewQueuedCategoricalPolicyQuickSetup.inputQueueArray, inputArray)
		
		local agentIndexQueueOutputArray = NewQueuedCategoricalPolicyQuickSetup.agentIndexOutputQueueArray

		local outputQueueArray = NewQueuedCategoricalPolicyQuickSetup.outputQueueArray
		
		local outputQueueArrayIndex
		
		repeat
			
			task.wait()
			
			outputQueueArrayIndex = table.find(agentIndexQueueOutputArray, agentIndex)
			
		until (outputQueueArrayIndex)
		
		local action, actionValue, actionVector, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue = table.unpack(outputQueueArray[outputQueueArrayIndex])
		
		table.remove(agentIndexQueueOutputArray, outputQueueArrayIndex)
		
		table.remove(outputQueueArray, outputQueueArrayIndex)

		previousActionDictionary[agentIndex] = action

		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector
		
		selectedActionCountVectorDictionary[selectedActionCountVectorIndex] = selectedActionCountVector
		
		currentEpsilonDictionary[currentEpsilonIndex] = currentEpsilon
		
		currentTemperatureDictionary[currentTemperature] = currentTemperature
		
		currentCValueDictionary[currentCValueIndex] = currentCValue
		
		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes
		
		if (NewQueuedCategoricalPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	NewQueuedCategoricalPolicyQuickSetup:setResetFunction(function()

		NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = {}

		NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary = {}

		NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = {}

		NewQueuedCategoricalPolicyQuickSetup.currentEpsilonDictionary = {}
		
		NewQueuedCategoricalPolicyQuickSetup.currentTemperatureictionary = {}
		
		NewQueuedCategoricalPolicyQuickSetup.currentEpsilonDictionary = {}
		
		NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary = {}

		NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary = {}

		for _, EpsilonValueScheduler in ipairs(NewQueuedCategoricalPolicyQuickSetup.EpsilonValueSchedulerDictionary) do EpsilonValueScheduler:reset() end
		
		for _, TemperatureValueScheduler in ipairs(NewQueuedCategoricalPolicyQuickSetup.TemperatureValueSchedulerDictionary) do TemperatureValueScheduler:reset() end
		
		for _, CValueValueScheduler in ipairs(NewQueuedCategoricalPolicyQuickSetup.CValueValueSchedulerDictionary) do CValueValueScheduler:reset() end

		for _, ExperienceReplay in ipairs(NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary) do ExperienceReplay:reset() end

		for _, EligibilityTrace in ipairs(NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary) do EligibilityTrace:reset() end
		
		NewQueuedCategoricalPolicyQuickSetup.inputQueueArray = {}

		NewQueuedCategoricalPolicyQuickSetup.agentIndexOutputQueueArray = {}

		NewQueuedCategoricalPolicyQuickSetup.outputQueueArray = {}

	end)
	
	return NewQueuedCategoricalPolicyQuickSetup
	
end

function QueuedCategoricalPolicyQuickSetup:start()
	
	if (self.isRunning) then error("It is already active.") end
	
	local functionToRun = coroutine.create(function()
		
		self.isRunning = true
		
		local Model = self.Model

		local numberOfReinforcementsPerEpisode = self.numberOfReinforcementsPerEpisode

		local updateFunction = self.updateFunction

		local episodeUpdateFunction = self.episodeUpdateFunction

		local inputQueueArray = self.inputQueueArray

		local agentIndexQueueOutputArray = self.agentIndexOutputQueueArray

		local outputQueueArray = self.outputQueueArray

		local ActionsList = Model:getActionsList()

		local agentIndex

		local previousFeatureVector

		local previousAction

		local rewardValue

		local currentFeatureVector

		local terminalStateValue

		local isEpisodeEnd

		local ExperienceReplay

		local EligibilityTrace
		
		local selectedActionCountVector
		
		local currentEpsilon
		
		local currentTemperature
		
		local currentCValue
		
		local EpsilonValueScheduler
		
		local TemperatureValueScheduler
		
		local CValueValueScheduler
		
		local currentNumberOfReinforcements

		local isOriginalValueNotAVector

		local currentActionVector

		local currentActionIndex

		local currentAction

		local currentActionValue

		local temporalDifferenceError

		local outputArray

		while(self.isRunning) do

			while (#inputQueueArray == 0) do task.wait() end
			
			pcall(function()
				
				agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, ExperienceReplay, EligibilityTrace, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue, EpsilonValueScheduler, TemperatureValueScheduler, CValueValueScheduler, currentNumberOfReinforcements = table.unpack(inputQueueArray[1])

				isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")

				if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end

				currentActionVector = Model:predict(currentFeatureVector, true)

				Model.EligibilityTrace = EligibilityTrace

				if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

				currentActionIndex, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue = self:selectAction(currentActionVector, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue, EpsilonValueScheduler, TemperatureValueScheduler, CValueValueScheduler, currentNumberOfReinforcements)

				currentAction = ActionsList[currentActionIndex]

				currentActionValue = currentActionVector[1][currentActionIndex]

				if (previousFeatureVector) then

					temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

					if (updateFunction) then updateFunction(terminalStateValue, agentIndex) end

				end

				if (isEpisodeEnd) then

					Model:episodeUpdate(terminalStateValue)

					if (episodeUpdateFunction) then episodeUpdateFunction(terminalStateValue, agentIndex) end

				end

				if (ExperienceReplay) and (previousFeatureVector) then

					ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

					ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

					ExperienceReplay:run(function(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

						return Model:categoricalUpdate(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

					end)

				end

				outputArray = {currentAction, currentActionValue, currentActionVector, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue}
				
				table.insert(outputQueueArray, outputArray)

				table.insert(agentIndexQueueOutputArray, agentIndex)
				
			end)

			table.remove(inputQueueArray, 1)

		end
		
	end)
	
	coroutine.resume(functionToRun)
	
end

function QueuedCategoricalPolicyQuickSetup:stop()
	
	if (not self.isRunning) then error("It is not active.") end
	
	self.isRunning = false
	
end

return QueuedCategoricalPolicyQuickSetup
