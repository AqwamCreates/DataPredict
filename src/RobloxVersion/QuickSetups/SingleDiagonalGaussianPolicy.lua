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

local ReinforcementLearningBaseQuickSetup = require(script.Parent.ReinforcementLearningBaseQuickSetup)

SingleDiagonalGaussianPolicyQuickSetup = {}

SingleDiagonalGaussianPolicyQuickSetup.__index = SingleDiagonalGaussianPolicyQuickSetup

setmetatable(SingleDiagonalGaussianPolicyQuickSetup, ReinforcementLearningBaseQuickSetup)

local defaultCurrentNumberOfReinforcements = 0

local defaultCurrentNumberOfEpisodes = 1

function SingleDiagonalGaussianPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSingleDiagonalGaussianPolicyQuickSetup = ReinforcementLearningBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewSingleDiagonalGaussianPolicyQuickSetup, SingleDiagonalGaussianPolicyQuickSetup)
	
	NewSingleDiagonalGaussianPolicyQuickSetup:setName("SingleDiagonalGaussianPolicyQuickSetup")
	
	NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements = parameterDictionary.currentNumberOfReinforcements or defaultCurrentNumberOfReinforcements

	NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes = parameterDictionary.currentNumberOfEpisodes or defaultCurrentNumberOfEpisodes
	
	NewSingleDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector = parameterDictionary.actionStandardDeviationVector
	
	NewSingleDiagonalGaussianPolicyQuickSetup.previousActionMeanVector = parameterDictionary.previousActionMeanVector

	NewSingleDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector = parameterDictionary.previousActionNoiseVector
	
	NewSingleDiagonalGaussianPolicyQuickSetup.ExperienceReplay = parameterDictionary.ExperienceReplay
	
	NewSingleDiagonalGaussianPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue)
		
		local Model = NewSingleDiagonalGaussianPolicyQuickSetup.Model

		if (not Model) then error("No model.") end
		
		local numberOfReinforcementsPerEpisode = NewSingleDiagonalGaussianPolicyQuickSetup.numberOfReinforcementsPerEpisode

		local currentNumberOfReinforcements = NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements + 1

		local currentNumberOfEpisodes = NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes
		
		local ExperienceReplay = NewSingleDiagonalGaussianPolicyQuickSetup.ExperienceReplay

		local previousFeatureVector = NewSingleDiagonalGaussianPolicyQuickSetup.previousFeatureVector

		local currentActionMeanVector = Model:predict(currentFeatureVector, true)
		
		local actionStandardDeviationVector = NewSingleDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector
		
		local previousActionNoiseVector = NewSingleDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector
		
		local previousActionMeanVector =  NewSingleDiagonalGaussianPolicyQuickSetup.previousActionMeanVector
		
		local actionVectorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(currentActionMeanVector)

		local currentActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor(actionVectorDimensionSizeArray, 0, 1)
		
		local currentScaledActionNoiseVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, currentActionNoiseVector)
		
		local actionVector = AqwamTensorLibrary:add(currentActionMeanVector, currentScaledActionNoiseVector)
		
		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)
		
		local terminalStateValue = 0
	
		local temporalDifferenceError
		
		if (isEpisodeEnd) then terminalStateValue = 1 end

		if (previousFeatureVector) then
			
			local updateFunction = NewSingleDiagonalGaussianPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:diagonalGaussianUpdate(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			local episodeUpdateFunction = NewSingleDiagonalGaussianPolicyQuickSetup.episodeUpdateFunction

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

		NewSingleDiagonalGaussianPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewSingleDiagonalGaussianPolicyQuickSetup.previousActionMeanVector = currentActionMeanVector
		
		NewSingleDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector = currentActionNoiseVector
		
		NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements = currentNumberOfReinforcements

		NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes
		
		if (NewSingleDiagonalGaussianPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end
		
		return actionVector
		
	end)
	
	NewSingleDiagonalGaussianPolicyQuickSetup:setResetFunction(function()
		
		NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements = 0

		NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes = 1

		NewSingleDiagonalGaussianPolicyQuickSetup.previousFeatureVector = nil

		NewSingleDiagonalGaussianPolicyQuickSetup.previousActionMeanVector = nil

		NewSingleDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector = nil
		
	end)
	
	return NewSingleDiagonalGaussianPolicyQuickSetup
	
end

return SingleDiagonalGaussianPolicyQuickSetup
