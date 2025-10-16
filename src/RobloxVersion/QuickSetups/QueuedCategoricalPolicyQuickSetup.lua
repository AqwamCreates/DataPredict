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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local CategoricalPolicyBaseQuickSetup = require(script.Parent.CategoricalPolicyBaseQuickSetup)

QueuedCategoricalPolicyQuickSetup = {}

QueuedCategoricalPolicyQuickSetup.__index = QueuedCategoricalPolicyQuickSetup

setmetatable(QueuedCategoricalPolicyQuickSetup, CategoricalPolicyBaseQuickSetup)

local defaultShareExperienceReplay = false

local defaultShareEligibilityTrace = false

local defaultShareSelectedActionCountVector = false

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
	
	NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)
	
	NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)
	
	-- Dictionaries

	NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary = parameterDictionary.EligibilityTraceDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary = parameterDictionary.previousActionDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary = parameterDictionary.selectedActionCountVectorDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}
	
	NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	-- Queues
	
	NewQueuedCategoricalPolicyQuickSetup.inputQueueArray = parameterDictionary.inputQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.agentIndexQueueOutputArray = parameterDictionary.agentIndexOutputQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.outputQueueArray = parameterDictionary.outputQueueArray or {}
	
	-- Debounce
	
	NewQueuedCategoricalPolicyQuickSetup.isRunning = false
	
	NewQueuedCategoricalPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue, returnOriginalOutput)
		
		if (not NewQueuedCategoricalPolicyQuickSetup.isRunning) then error("Not currently running.") end
		
		local experienceReplayIndex = (NewQueuedCategoricalPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex
		
		local eligibilityTraceIndex = (NewQueuedCategoricalPolicyQuickSetup.shareEligibilityTrace and 1) or agentIndex
		
		local selectedActionCountVectorIndex = (NewQueuedCategoricalPolicyQuickSetup.shareSelectedActionCountVector and 1) or agentIndex
		
		local numberOfReinforcementsIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex
		
		local numberOfEpisodesIndex = (NewQueuedCategoricalPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex
		
		local previousFeatureVectorDictionary = NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary
		
		local previousActionDictionary = NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary
		
		local selectedActionCountVectorDictionary = NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVectorDictionary
		
		local currentNumberOfReinforcementsDictionary = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcementsDictionary
		
		local currentNumberOfEpisodesDictionary = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodesDictionary
		
		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]
		
		local previousAction = previousActionDictionary[agentIndex]
		
		local selectedActionCountVector = selectedActionCountVectorDictionary[selectedActionCountVectorIndex]
		
		local ExperienceReplay = NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]
		
		local EligibilityTrace = NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary[eligibilityTraceIndex]
		
		local currentNumberOfReinforcements = currentNumberOfReinforcementsDictionary[numberOfReinforcementsIndex] or 0
		
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
		
		local inputArray = {agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, selectedActionCountVector, ExperienceReplay, EligibilityTrace}
		
		table.insert(NewQueuedCategoricalPolicyQuickSetup.inputQueueArray, inputArray)
		
		local agentIndexQueueOutputArray = parameterDictionary.agentIndexOutputQueueArray

		local outputQueueArray = parameterDictionary.outputQueueArray
		
		local outputQueueArrayIndex
		
		repeat
			
			task.wait()
			
			outputQueueArrayIndex = table.find(agentIndexQueueOutputArray, agentIndex)
			
		until (outputQueueArrayIndex)
		
		local action, actionValue, actionVector, selectedActionCountVector = table.unpack(outputQueueArray[outputQueueArrayIndex])
		
		table.remove(agentIndexQueueOutputArray, outputQueueArrayIndex)
		
		table.remove(outputQueueArray, outputQueueArrayIndex)

		previousActionDictionary[agentIndex] = action

		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes

		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector
		
		selectedActionCountVectorDictionary[selectedActionCountVectorIndex] = selectedActionCountVector
		
		if (NewQueuedCategoricalPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	return NewQueuedCategoricalPolicyQuickSetup
	
end

function QueuedCategoricalPolicyQuickSetup:start()
	
	if (self.isRunning) then error("It is already active.") end
	
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
	
	local selectedActionCountVector
	
	local ExperienceReplay

	local EligibilityTrace
	
	local isEpisodeEnd
	
	local isOriginalValueNotAVector
	
	local actionVector
	
	local actionIndex
	
	local action
	
	local actionValue
	
	local temporalDifferenceError
	
	local outputArray
	
	while(self.isRunning) do
		
		while (#inputQueueArray == 0) do task.wait() end
		
		agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, selectedActionCountVector, ExperienceReplay, EligibilityTrace = table.unpack(inputQueueArray[1])
		
		isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")

		if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end

		actionVector = Model:predict(currentFeatureVector, true)

		terminalStateValue = 0

		Model.EligibilityTrace = EligibilityTrace

		if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

		actionIndex, selectedActionCountVector = self:selectAction(actionVector, selectedActionCountVector)

		action = ActionsList[actionIndex]

		actionValue = actionVector[1][actionIndex]

		if (previousFeatureVector) then

			temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue, agentIndex) end

		end

		if (isEpisodeEnd) then

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue, agentIndex) end

		end

		if (previousFeatureVector) then

			if (ExperienceReplay) then

				ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

				ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

				ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

					return Model:categoricalUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

				end)

			end

		end

		outputArray = {action, actionValue, actionVector, selectedActionCountVector}

		table.remove(inputQueueArray, 1)

		table.insert(outputQueueArray, outputArray)

		table.insert(agentIndexQueueOutputArray, agentIndex)
		
	end
	
end

function QueuedCategoricalPolicyQuickSetup:stop()
	
	if (not self.isRunning) then error("It is not active.") end
	
	self.isRunning = false
	
end

function QueuedCategoricalPolicyQuickSetup:reset()
	
	self.previousFeatureVectorDictionary = {}

	self.previousActionDictionary = {}
	
	self.selectedActionCountVectorDictionary = {}

	self.currentNumberOfReinforcementsDictionary  = {}

	self.currentNumberOfEpisodesDictionary  = {}
	
	for _, ExperienceReplay in ipairs(self.ExperienceReplayDictionary) do ExperienceReplay:reset() end
	
	for _, EligibilityTrace in ipairs(self.EligibilityTraceDictionary) do EligibilityTrace:reset() end
		
end

return QueuedCategoricalPolicyQuickSetup
