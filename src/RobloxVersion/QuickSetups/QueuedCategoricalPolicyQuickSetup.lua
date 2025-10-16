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

local DiagonalGaussianPolicyBaseQuickSetup = require(script.Parent.DiagonalGaussianPolicyBaseQuickSetup)

QueuedDiagonalGaussianPolicyQuickSetup = {}

QueuedDiagonalGaussianPolicyQuickSetup.__index = QueuedDiagonalGaussianPolicyQuickSetup

setmetatable(QueuedDiagonalGaussianPolicyQuickSetup, DiagonalGaussianPolicyBaseQuickSetup)

local defaultShareExperienceReplay = false

local defaultShareCurrentNumberOfReinforcements = false

local defaultShareCurrentNumberOfEpisodes = false

function QueuedDiagonalGaussianPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewQueuedDiagonalGaussianPolicyQuickSetup = DiagonalGaussianPolicyBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewQueuedDiagonalGaussianPolicyQuickSetup, QueuedDiagonalGaussianPolicyQuickSetup)
	
	NewQueuedDiagonalGaussianPolicyQuickSetup:setName("QueuedDiagonalGaussianPolicyQuickSetup")
	
	-- Share toggles
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.shareExperienceReplay = NewQueuedDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewQueuedDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewQueuedDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)
	
	-- Dictionaries

	NewQueuedDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewQueuedDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionDictionary = parameterDictionary.previousActionDictionary or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	-- Queues
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.inputQueueArray = parameterDictionary.inputQueueArray or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.agentIndexOutputQueueArray = parameterDictionary.agentIndexOutputQueueArray or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.outputQueueArray = parameterDictionary.outputQueueArray or {}
	
	-- Debounce
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.isRunning = false
	
	NewQueuedDiagonalGaussianPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue, returnOriginalOutput)
		
		if (not NewQueuedDiagonalGaussianPolicyQuickSetup.isRunning) then error("Not currently running.") end
		
		local experienceReplayIndex = (NewQueuedDiagonalGaussianPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex
		
		local numberOfReinforcementsIndex = (NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex
		
		local numberOfEpisodesIndex = (NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex
		
		local previousFeatureVectorDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary
		
		local previousActionDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionDictionary
		
		local currentNumberOfReinforcementsDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary
		
		local currentNumberOfEpisodesDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary
		
		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]
		
		local previousAction = previousActionDictionary[agentIndex]
		
		local ExperienceReplay = NewQueuedDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]
		
		local currentNumberOfReinforcements = currentNumberOfReinforcementsDictionary[numberOfReinforcementsIndex] or 0
		
		local currentNumberOfEpisodes = currentNumberOfEpisodesDictionary[numberOfEpisodesIndex] or 1
		
		local terminalStateValue 
		
		local isEpisodeEnd
		
		if (currentNumberOfReinforcements >= NewQueuedDiagonalGaussianPolicyQuickSetup.numberOfReinforcementsPerEpisode) then
			
			isEpisodeEnd = true
			
			terminalStateValue = 1

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			currentNumberOfReinforcements = 0

		else
			
			isEpisodeEnd = false
			
			terminalStateValue = 0

			currentNumberOfReinforcements = currentNumberOfReinforcements + 1

		end
		
		local inputArray = {agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, ExperienceReplay, currentNumberOfReinforcements}
		
		table.insert(NewQueuedDiagonalGaussianPolicyQuickSetup.inputQueueArray, inputArray)
		
		local agentIndexQueueOutputArray = NewQueuedDiagonalGaussianPolicyQuickSetup.agentIndexOutputQueueArray

		local outputQueueArray = NewQueuedDiagonalGaussianPolicyQuickSetup.outputQueueArray
		
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
		
		if (NewQueuedDiagonalGaussianPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	return NewQueuedDiagonalGaussianPolicyQuickSetup
	
end

function QueuedDiagonalGaussianPolicyQuickSetup:start()
	
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
		
		local currentNumberOfReinforcements

		local isOriginalValueNotAVector

		local actionVector

		local temporalDifferenceError

		local outputArray

		while(self.isRunning) do

			while (#inputQueueArray == 0) do task.wait() end

			agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, ExperienceReplay, currentNumberOfReinforcements = table.unpack(inputQueueArray[1])

			isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")

			if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end

			actionVector = Model:predict(currentFeatureVector, true)

			terminalStateValue = 0

			if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

			if (previousFeatureVector) then

				temporalDifferenceError = Model:DiagonalGaussianUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

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

						return Model:DiagonalGaussianUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

					end)

				end

			end

			outputArray = {actionVector}

			table.remove(inputQueueArray, 1)

			table.insert(outputQueueArray, outputArray)

			table.insert(agentIndexQueueOutputArray, agentIndex)

		end
		
	end)
	
	coroutine.resume(functionToRun)
	
end

function QueuedDiagonalGaussianPolicyQuickSetup:stop()
	
	if (not self.isRunning) then error("It is not active.") end
	
	self.isRunning = false
	
end

function QueuedDiagonalGaussianPolicyQuickSetup:reset()
	
	self.previousFeatureVectorDictionary = {}

	self.previousActionDictionary = {}

	self.currentNumberOfReinforcementsDictionary  = {}

	self.currentNumberOfEpisodesDictionary  = {}
	
	for _, ExperienceReplay in ipairs(self.ExperienceReplayDictionary) do ExperienceReplay:reset() end
		
end

return QueuedDiagonalGaussianPolicyQuickSetup
