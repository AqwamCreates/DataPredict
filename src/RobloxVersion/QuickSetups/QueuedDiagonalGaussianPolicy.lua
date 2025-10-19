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
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector = parameterDictionary.actionStandardDeviationVector
	
	-- Share toggles
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.shareExperienceReplay = NewQueuedDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewQueuedDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewQueuedDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)
	
	-- Dictionaries

	NewQueuedDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewQueuedDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionMeanVectorDictionary = parameterDictionary.previousActionMeanVectorDictionary or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionNoiseVectorDictionary = parameterDictionary.previousActionNoiseVectorDictionary or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	-- Queues
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.inputQueueArray = parameterDictionary.inputQueueArray or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.agentIndexOutputQueueArray = parameterDictionary.agentIndexOutputQueueArray or {}
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.outputQueueArray = parameterDictionary.outputQueueArray or {}
	
	-- Debounce
	
	NewQueuedDiagonalGaussianPolicyQuickSetup.isRunning = false
	
	NewQueuedDiagonalGaussianPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue)
		
		if (not NewQueuedDiagonalGaussianPolicyQuickSetup.isRunning) then error("Not currently running.") end
		
		local experienceReplayIndex = (NewQueuedDiagonalGaussianPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex
		
		local numberOfReinforcementsIndex = (NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex
		
		local numberOfEpisodesIndex = (NewQueuedDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex
		
		local previousFeatureVectorDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary
		
		local previousActionMeanVectorDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionMeanVectorDictionary
		
		local previousActionNoiseVectorDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionNoiseVectorDictionary
		
		local currentNumberOfReinforcementsDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary
		
		local currentNumberOfEpisodesDictionary = NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary
		
		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]
		
		local previousActionMeanVector = previousActionMeanVectorDictionary[agentIndex]
		
		local previousActionNoiseVector = previousActionNoiseVectorDictionary[agentIndex]
		
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
		
		local inputArray = {agentIndex, previousFeatureVector, previousActionMeanVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, ExperienceReplay}
		
		table.insert(NewQueuedDiagonalGaussianPolicyQuickSetup.inputQueueArray, inputArray)
		
		local agentIndexQueueOutputArray = NewQueuedDiagonalGaussianPolicyQuickSetup.agentIndexOutputQueueArray

		local outputQueueArray = NewQueuedDiagonalGaussianPolicyQuickSetup.outputQueueArray
		
		local outputQueueArrayIndex
		
		repeat
			
			task.wait()
			
			outputQueueArrayIndex = table.find(agentIndexQueueOutputArray, agentIndex)
			
		until (outputQueueArrayIndex)
		
		local actionVector, currentActionMeanVector, currentActionNoiseVector = table.unpack(outputQueueArray[outputQueueArrayIndex])
		
		table.remove(agentIndexQueueOutputArray, outputQueueArrayIndex)
		
		table.remove(outputQueueArray, outputQueueArrayIndex)
		
		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector

		previousActionMeanVectorDictionary[agentIndex] = currentActionMeanVector
		
		previousActionNoiseVectorDictionary[agentIndex] = currentActionNoiseVector

		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes
		
		if (NewQueuedDiagonalGaussianPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		return actionVector
		
	end)
	
	NewQueuedDiagonalGaussianPolicyQuickSetup:setResetFunction(function(agentIndex, currentFeatureVector, rewardValue)

		NewQueuedDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary = {}

		NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionMeanVectorDictionary = {}

		NewQueuedDiagonalGaussianPolicyQuickSetup.previousActionNoiseVectorDictionary = {}

		NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary  = {}

		NewQueuedDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary  = {}

		for _, ExperienceReplay in ipairs(NewQueuedDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary) do ExperienceReplay:reset() end

	end)
	
	return NewQueuedDiagonalGaussianPolicyQuickSetup
	
end

function QueuedDiagonalGaussianPolicyQuickSetup:start()
	
	if (self.isRunning) then error("It is already active.") end
	
	local functionToRun = coroutine.create(function()
		
		self.isRunning = true
		
		local Model = self.Model
		
		local actionStandardDeviationVector = self.actionStandardDeviationVector

		local numberOfReinforcementsPerEpisode = self.numberOfReinforcementsPerEpisode

		local updateFunction = self.updateFunction

		local episodeUpdateFunction = self.episodeUpdateFunction

		local inputQueueArray = self.inputQueueArray

		local agentIndexQueueOutputArray = self.agentIndexOutputQueueArray

		local outputQueueArray = self.outputQueueArray

		local ActionsList = Model:getActionsList()

		local agentIndex

		local previousFeatureVector

		local previousActionMeanVector
		
		local previousActionNoiseVector

		local rewardValue

		local currentFeatureVector

		local terminalStateValue

		local isEpisodeEnd

		local ExperienceReplay
		
		local currentNumberOfReinforcements

		local isOriginalValueNotAVector

		local actionMeanVector

		local temporalDifferenceError
		
		local actionVectorDimensionSizeArray
		
		local actionVector
		
		local actionMeanVector
		
		local actionNoiseVector
		
		local scaledActionNoiseVector 
		
		local outputArray = {}

		while(self.isRunning) do

			while (#inputQueueArray == 0) do task.wait() end
			
			pcall(function()
				
				agentIndex, previousFeatureVector, previousActionMeanVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue, isEpisodeEnd, ExperienceReplay = table.unpack(inputQueueArray[1])

				isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")

				if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end

				actionMeanVector = Model:predict(currentFeatureVector, true)

				actionVectorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanVector)

				actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor(actionVectorDimensionSizeArray, 0, 1)

				scaledActionNoiseVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

				actionVector = AqwamTensorLibrary:add(actionMeanVector, scaledActionNoiseVector)

				if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

				if (previousFeatureVector) then

					temporalDifferenceError = Model:diagonalGaussianUpdate(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

					if (updateFunction) then updateFunction(terminalStateValue, agentIndex) end

				end

				if (isEpisodeEnd) then

					Model:episodeUpdate(terminalStateValue)

					if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue, agentIndex) end

				end

				if (ExperienceReplay) and (previousFeatureVector) then

					ExperienceReplay:addExperience(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

					ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

					ExperienceReplay:run(function(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

						return Model:diagonalGaussianUpdate(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

					end)

				end
				
				outputArray = {actionVector, actionMeanVector, actionNoiseVector}
				
				table.insert(outputQueueArray, outputArray)

				table.insert(agentIndexQueueOutputArray, agentIndex)
				
			end)

			table.remove(inputQueueArray, 1)

		end
		
	end)
	
	coroutine.resume(functionToRun)
	
end

function QueuedDiagonalGaussianPolicyQuickSetup:stop()
	
	if (not self.isRunning) then error("It is not active.") end
	
	self.isRunning = false
	
end

return QueuedDiagonalGaussianPolicyQuickSetup
