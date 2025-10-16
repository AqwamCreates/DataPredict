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
	
	-- Share toggles
	
	NewQueuedCategoricalPolicyQuickSetup.shareExperienceReplay =  NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)
	
	NewQueuedCategoricalPolicyQuickSetup.shareEligibilityTrace =  NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareEligibilityTrace or defaultShareEligibilityTrace)
	
	NewQueuedCategoricalPolicyQuickSetup.shareSelectedActionCountVector =  NewQueuedCategoricalPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareSelectedActionCountVector or defaultShareSelectedActionCountVector)
	
	
	-- Dictionaries

	NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary = parameterDictionary.EligibilityTraceDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary = parameterDictionary.previousActionDictionary or {}
	
	-- Queues
	
	NewQueuedCategoricalPolicyQuickSetup.informationQueueArray = parameterDictionary.informationQueueArray or {}
	
	NewQueuedCategoricalPolicyQuickSetup.isRunning = false
	
	NewQueuedCategoricalPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue, agentIndex, returnOriginalOutput)
		
		if (not NewQueuedCategoricalPolicyQuickSetup.isRunning) then error("Not currently running.") end
		
		local experienceReplayIndex = (NewQueuedCategoricalPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex
		
		local eligibilityTraceIndex = (NewQueuedCategoricalPolicyQuickSetup.shareEligibilityTrace and 1) or agentIndex
		
		local shareSelectedActionCountVectorIndex = (NewQueuedCategoricalPolicyQuickSetup.shareSelectedActionCountVector and 1) or agentIndex
		
		local previousFeatureVector = NewQueuedCategoricalPolicyQuickSetup.previousFeatureVectorDictionary[agentIndex]
		
		local previousAction = NewQueuedCategoricalPolicyQuickSetup.previousActionDictionary[agentIndex]
		
		local selectedActionCountVector = NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVector[shareSelectedActionCountVectorIndex]
		
		local ExperienceReplay = NewQueuedCategoricalPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]
		
		local EligibilityTrace = NewQueuedCategoricalPolicyQuickSetup.EligibilityTraceDictionary[eligibilityTraceIndex]
		
		local informationArray = {previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, selectedActionCountVector, ExperienceReplay, EligibilityTrace}
		
		table.insert(NewQueuedCategoricalPolicyQuickSetup.informationQueueArray, informationArray)
		
		

		NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcements = NewQueuedCategoricalPolicyQuickSetup.currentNumberOfReinforcements + 1

		NewQueuedCategoricalPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes

		NewQueuedCategoricalPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewQueuedCategoricalPolicyQuickSetup.previousAction = action
		
		NewQueuedCategoricalPolicyQuickSetup.selectedActionCountVector = selectedActionCountVector
		
		
		
		previousFeatureVector[agentIndex] = currentFeatureVector
		
		actionVectorArray[agentIndex] = action
		
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
	
	local numberOfReinforcementsPerEpisode = self.numberOfReinforcementsPerEpisode
	
	local updateFunction = self.updateFunction
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	local informationQueueArray = self.informationQueueArray
	
	local ActionsList = Model:getActionsList()
	
	local agentIndex
	
	local ExperienceReplay
	
	local EligibilityTrace
	
	local previousFeatureVector
	
	local previousAction
	
	local rewardValue
	
	local currentFeatureVector
	
	local selectedActionCountVector
	
	local isOriginalValueNotAVector
	
	local actionVector
	
	local actionIndex
	
	local action
	
	local actionValue
	
	local temporalDifferenceError
	
	while(self.isRunning) do
		
		agentIndex, previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue, selectedActionCountVector, ExperienceReplay, EligibilityTrace, currentNumberOfReinforcements, currentNumberOfEpisodes = table.unpack(informationQueueArray[1])
		
		if (agentIndex) then
			
			isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")

			if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end
			
			actionVector = Model:predict(currentFeatureVector, true)

			terminalStateValue = 0
			
			Model.EligibilityTrace = EligibilityTrace

			if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end

			actionIndex, selectedActionCountVector = self:selectAction(actionVector, selectedActionCountVector)

			action = ActionsList[actionIndex]

			actionValue = actionVector[1][actionIndex]

			if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then terminalStateValue = 1 end

			if (previousFeatureVector) then

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

			if (previousFeatureVector) then

				if (ExperienceReplay) then

					ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

					ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

					ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

						return Model:categoricalUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

					end)

				end

			end
			
			table.remove(informationQueueArray, 1)
			
		end
		
	end
	
end

function QueuedCategoricalPolicyQuickSetup:stop()
	
	self.isRunning = false
	
end

return QueuedCategoricalPolicyQuickSetup
