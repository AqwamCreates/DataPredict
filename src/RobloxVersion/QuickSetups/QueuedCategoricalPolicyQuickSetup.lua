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

function QueuedCategoricalPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewQueuedCategoricalPolicyQuickSetup = CategoricalPolicyBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewQueuedCategoricalPolicyQuickSetup, QueuedCategoricalPolicyQuickSetup)
	
	NewQueuedCategoricalPolicyQuickSetup:setName("QueuedCategoricalPolicyQuickSetup")
	
	NewQueuedCategoricalPolicyQuickSetup.shareExperienceReplay =  NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)
	
	NewQueuedCategoricalPolicyQuickSetup.shareEligibilityTrace =  NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareEligibilityTrace or defaultShareEligibilityTrace)
	
	NewQueuedCategoricalPolicyQuickSetup.shareSelectedActionCountVector =  NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareSelectedActionCountVector or defaultShareSelectedActionCountVector)
	
	NewQueuedCategoricalPolicyQuickSetup.agentIndexQueueArray = parameterDictionary.agentIndexQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayQueueArray = parameterDictionary.ExperienceReplayQueueArray or {}

	NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceQueueArray = parameterDictionary.EligibilityTraceQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.currentFeatureVectorQueueArray = parameterDictionary.currentFeatureVectorQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.rewardValueQueueArray = parameterDictionary.rewardValueQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.previousActionArray = parameterDictionary.previousActionArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVectorArray = parameterDictionary.selectedActionCountVectorArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.actionVectorArray = parameterDictionary.actionVectorArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.isRunning = false
	
	NewQueuedCategoricalPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue, agentIndex, returnOriginalOutput)
		
		if (not NewQueuedCategoricalPolicyQuickSetup.isRunning) then error("Not currently running.") end
		
		local actionVectorArray = NewQueuedCategoricalPolicyQuickSetup.actionVectorArray
		
		table.insert(NewQueuedCategoricalPolicyQuickSetup.currentFeatureVectorQueueArray, currentFeatureVector)
		
		table.insert(NewQueuedCategoricalPolicyQuickSetup.agentIndexQueueArray, agentIndex)
		
		

		NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcements = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcements + 1

		NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes

		NewQueuedCategoricalPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewQueuedCategoricalPolicyQuickSetup.previousAction = action
		
		NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVector = selectedActionCountVector
		
		
		
		actionVectorArray[agentIndex] = nil
		
		if (NewQueuedCategoricalPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	return NewQueuedCategoricalPolicyQuickSetup
	
end

function QueuedCategoricalPolicyQuickSetup:start()
	
	if (self.isRunning) then error("It is already running.") end
	
	self.isRunning = true
	
	local Model = self.Model

	local shareExperienceReplay =  self.shareExperienceReplay

	local shareEligibilityTrace =  self.shareEligibilityTrace
	
	local numberOfReinforcementsPerEpisode = self.numberOfReinforcementsPerEpisode
	
	local agentIndexQueueArray = self.agentIndexQueueArray
	
	local ExperienceReplayQueueArray = self.ExperienceReplayQueueArray

	local EligibilityTraceQueueArray = self.EligibilityTraceQueueArray
	
	local currentFeatureVectorQueueArray = self.currentFeatureVectorQueueArray

	local rewardValueQueueArray = self.rewardValueQueueArray
	
	local previousActionArray = self.previousActionArray

	local selectedActionCountVectorArray = self.selectedActionCountVectorArray
	
	local ActionsList = Model:getActionsList()
	
	local updateFunction = self.updateFunction
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	local agentIndex
	
	local ExperienceReplay
	
	local EligibilityTrace
	
	local previousFeatureVector
	
	local previousAction
	
	local rewardValue
	
	local currentFeatureVector
	
	local selectedActionCountVector
	
	local isOriginalValueNotAVector
	
	local hasPreviousFeatureVector
	
	local actionIndex
	
	local selectedActionCountVector
	
	local action
	
	local actionValue
	
	while(self.isRunning) do
		
		agentIndex = agentIndexQueueArray[1]
		
		ExperienceReplay = ExperienceReplayQueueArray[1]
		
		EligibilityTrace = EligibilityTraceQueueArray[1]
		
		previousFeatureVector = previousActionArray[1]
		
		previousAction = previousActionArray[1]
		
		rewardValue = rewardValueQueueArray[1]
		
		currentFeatureVector = currentFeatureVectorQueueArray[1]
		
		selectedActionCountVector = selectedActionCountVector[1]
		
		if (agentIndex) then
			
			hasPreviousFeatureVector = (type(hasPreviousFeatureVector) == "table") -- To avoide reading the other agents' previous feature vectors.
			
			isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")

			if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end

			local currentNumberOfReinforcements = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcements

			local currentNumberOfEpisodes = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodes

			local previousAction = previousActionArray[agentIndex]
			
			local actionVector = Model:predict(currentFeatureVector, true)

			local terminalStateValue = 0

			local temporalDifferenceError
			
			Model.EligibilityTrace = EligibilityTrace

			if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

			actionIndex, selectedActionCountVector = self:selectAction(actionVector, selectedActionCountVectorArray[agentIndex])

			action = ActionsList[actionIndex]

			actionValue = actionVector[1][actionIndex]

			if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then terminalStateValue = 1 end

			if (hasPreviousFeatureVector) then

				currentNumberOfReinforcements = currentNumberOfReinforcements + 1

				temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

				if (updateFunction) then updateFunction(terminalStateValue, agentIndex) end

			end

			if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then

				currentNumberOfReinforcements = 0

				currentNumberOfEpisodes = currentNumberOfEpisodes + 1

				Model:episodeUpdate(terminalStateValue)

				if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue, agentIndex) end

			end

			if (hasPreviousFeatureVector) then

				if (ExperienceReplay) then

					ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

					ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

					ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

						return Model:categoricalUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

					end)

				end

			end
			
			table.remove(agentIndexQueueArray, 1)
			
			table.remove(ExperienceReplayQueueArray, 1)
			
			table.remove(EligibilityTraceQueueArray, 1)
			
			table.remove(previousFeatureVector, 1)
			
			table.remove(previousActionArray, 1)
			
			table.remove(rewardValueQueueArray, 1)
			
			table.remove(currentFeatureVectorQueueArray, 1)
			
			table.remove(selectedActionCountVectorArray, 1)
			
		end
		
	end
	
end

function QueuedCategoricalPolicyQuickSetup:stop()
	
	self.isRunning = false
	
end

return QueuedCategoricalPolicyQuickSetup
