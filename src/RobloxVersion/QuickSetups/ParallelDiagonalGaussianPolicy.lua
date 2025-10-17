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

ParallelDiagonalGaussianPolicyQuickSetup = {}

ParallelDiagonalGaussianPolicyQuickSetup.__index = ParallelDiagonalGaussianPolicyQuickSetup

setmetatable(ParallelDiagonalGaussianPolicyQuickSetup, DiagonalGaussianPolicyBaseQuickSetup)

local defaultShareExperienceReplay = false

local defaultShareCurrentNumberOfReinforcements = false

local defaultShareCurrentNumberOfEpisodes = false

function ParallelDiagonalGaussianPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewParallelDiagonalGaussianPolicyQuickSetup = DiagonalGaussianPolicyBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewParallelDiagonalGaussianPolicyQuickSetup, ParallelDiagonalGaussianPolicyQuickSetup)
	
	NewParallelDiagonalGaussianPolicyQuickSetup:setName("ParallelDiagonalGaussianPolicyQuickSetup")
	
	NewParallelDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector = parameterDictionary.actionStandardDeviationVector
	
	-- Share toggles
	
	NewParallelDiagonalGaussianPolicyQuickSetup.shareExperienceReplay = NewParallelDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareExperienceReplay or defaultShareExperienceReplay)
	
	NewParallelDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfReinforcements = NewParallelDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfReinforcements or defaultShareCurrentNumberOfReinforcements)
	
	NewParallelDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfEpisodes = NewParallelDiagonalGaussianPolicyQuickSetup:getValueOrDefaultValue(parameterDictionary.shareCurrentNumberOfEpisodes or defaultShareCurrentNumberOfEpisodes)
	
	-- Dictionaries

	NewParallelDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary = parameterDictionary.ExperienceReplayDictionary or {}

	NewParallelDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary = parameterDictionary.previousFeatureVectorDictionary or {}

	NewParallelDiagonalGaussianPolicyQuickSetup.previousActionMeanVectorDictionary = parameterDictionary.previousActionMeanVectorDictionary or {}
	
	NewParallelDiagonalGaussianPolicyQuickSetup.previousActionNoiseVectorDictionary = parameterDictionary.previousActionNoiseVectorDictionary or {}
	
	NewParallelDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary = parameterDictionary.currentNumberOfReinforcementsDictionary or {}
	
	NewParallelDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary = parameterDictionary.currentNumberOfEpisodesDictionary or {}
	
	NewParallelDiagonalGaussianPolicyQuickSetup:setReinforceFunction(function(agentIndex, currentFeatureVector, rewardValue)
		
		local Model = NewParallelDiagonalGaussianPolicyQuickSetup.Model

		if (not Model) then error("No model.") end
		
		local actionStandardDeviationVector = NewParallelDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector
		
		local numberOfReinforcementsPerEpisode = NewParallelDiagonalGaussianPolicyQuickSetup.numberOfReinforcementsPerEpisode
		
		local experienceReplayIndex = (NewParallelDiagonalGaussianPolicyQuickSetup.shareExperienceReplay and 1) or agentIndex
		
		local numberOfReinforcementsIndex = (NewParallelDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfReinforcements and 1) or agentIndex
		
		local numberOfEpisodesIndex = (NewParallelDiagonalGaussianPolicyQuickSetup.shareCurrentNumberOfEpisodes and 1) or agentIndex
		
		local previousFeatureVectorDictionary = NewParallelDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary
		
		local previousActionMeanVectorDictionary = NewParallelDiagonalGaussianPolicyQuickSetup.previousActionMeanVectorDictionary
		
		local previousActionNoiseVectorDictionary = NewParallelDiagonalGaussianPolicyQuickSetup.previousActionNoiseVectorDictionary
		
		local currentNumberOfReinforcementsDictionary = NewParallelDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary
		
		local currentNumberOfEpisodesDictionary = NewParallelDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary
		
		local previousFeatureVector = previousFeatureVectorDictionary[agentIndex]
		
		local previousActionMeanVector = previousActionMeanVectorDictionary[agentIndex]
		
		local previousActionNoiseVector = previousActionNoiseVectorDictionary[agentIndex]
		
		local ExperienceReplay = NewParallelDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary[experienceReplayIndex]
		
		local currentNumberOfReinforcements = currentNumberOfReinforcementsDictionary[numberOfReinforcementsIndex] or 0
		
		local currentNumberOfEpisodes = currentNumberOfEpisodesDictionary[numberOfEpisodesIndex] or 1
		
		local currentActionMeanVector = Model:predict(currentFeatureVector, true)
		
		local actionVectorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(currentActionMeanVector)

		local currentActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor(actionVectorDimensionSizeArray, 0, 1)

		local currentScaledActionNoiseVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, currentActionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(currentActionMeanVector, currentScaledActionNoiseVector)

		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)

		local terminalStateValue = 0

		local temporalDifferenceError

		if (isEpisodeEnd) then terminalStateValue = 1 end

		if (previousFeatureVector) then

			local updateFunction = NewParallelDiagonalGaussianPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:diagonalGaussianUpdate(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			local episodeUpdateFunction = NewParallelDiagonalGaussianPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue) end

		end

		if (ExperienceReplay) and (previousFeatureVector) then

			ExperienceReplay:addExperience(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

				return Model:diagonalGaussianUpdate(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

			end)

		end

		previousActionMeanVectorDictionary[agentIndex] = currentActionMeanVector
		
		previousActionNoiseVectorDictionary[agentIndex] = currentActionNoiseVector

		currentNumberOfReinforcementsDictionary[agentIndex] = currentNumberOfReinforcements

		currentNumberOfEpisodesDictionary[agentIndex] = currentNumberOfEpisodes

		previousFeatureVectorDictionary[agentIndex] = currentFeatureVector
		
		if (NewParallelDiagonalGaussianPolicyQuickSetup.isOutputPrinted) then
			
			print("Agent index: " .. agentIndex .. "\t\tEpisode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) 
			
		end

		return actionVector
		
	end)
	
	NewParallelDiagonalGaussianPolicyQuickSetup:setResetFunction(function(agentIndex, currentFeatureVector, rewardValue)
		
		ParallelDiagonalGaussianPolicyQuickSetup.previousFeatureVectorDictionary = {}

		ParallelDiagonalGaussianPolicyQuickSetup.previousActionMeanVectorDictionary = {}

		ParallelDiagonalGaussianPolicyQuickSetup.previousActionNoiseVectorDictionary = {}

		ParallelDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcementsDictionary  = {}

		ParallelDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodesDictionary  = {}

		for _, ExperienceReplay in ipairs(ParallelDiagonalGaussianPolicyQuickSetup.ExperienceReplayDictionary) do ExperienceReplay:reset() end
		
	end)
	
	return NewParallelDiagonalGaussianPolicyQuickSetup
	
end

return ParallelDiagonalGaussianPolicyQuickSetup
