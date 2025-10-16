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

local defaultCurrentNumberOfEpisodes = 0

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
	
	NewSingleDiagonalGaussianPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue)
		
		local Model = NewSingleDiagonalGaussianPolicyQuickSetup.Model

		if (not Model) then error("No model.") end
		
		local numberOfReinforcementsPerEpisode = NewSingleDiagonalGaussianPolicyQuickSetup.numberOfReinforcementsPerEpisode

		local currentNumberOfReinforcements = NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements + 1

		local currentNumberOfEpisodes = NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes
		
		local ExperienceReplay = NewSingleDiagonalGaussianPolicyQuickSetup.ExperienceReplay

		local previousFeatureVector = NewSingleDiagonalGaussianPolicyQuickSetup.previousFeatureVector

		local actionMeanVector = Model:predict(currentFeatureVector, true)
		
		local actionStandardDeviationVector = NewSingleDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector
		
		local previousActionMeanVector =  NewSingleDiagonalGaussianPolicyQuickSetup.previousActionMeanVector

		local previousActionNoiseVector = NewSingleDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector
		
		local actionVectorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanVector)

		local actionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor(actionVectorDimensionSizeArray)
		
		local actionVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)
		
		local terminalStateValue = 0
	
		local temporalDifferenceError
		
		if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then terminalStateValue = 1 end

		if (previousFeatureVector) then
			
			local updateFunction = NewSingleDiagonalGaussianPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:diagonalGaussianUpdate(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then

			local episodeUpdateFunction = NewSingleDiagonalGaussianPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue) end

		end
		
		if (ExperienceReplay) and (previousFeatureVector) then

			ExperienceReplay:addExperience(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storeTerminalStateValue)

				return Model:diagonalGaussianUpdate(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storeTerminalStateValue)

			end)

		end

		NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements = currentNumberOfReinforcements

		NewSingleDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes

		NewSingleDiagonalGaussianPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewSingleDiagonalGaussianPolicyQuickSetup.previousActionMeanVector = actionMeanVector
		
		NewSingleDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector = actionNoiseVector
		
		if (NewSingleDiagonalGaussianPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end
		
		return actionVector
		
	end)
	
	return NewSingleDiagonalGaussianPolicyQuickSetup
	
end

return SingleDiagonalGaussianPolicyQuickSetup
