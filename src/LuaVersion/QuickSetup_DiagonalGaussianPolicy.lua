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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local ReinforcementLearningBaseQuickSetup = require("QuickSetup_ReinforcementLearningBaseQuickSetup")

DiagonalGaussianPolicyQuickSetup = {}

DiagonalGaussianPolicyQuickSetup.__index = DiagonalGaussianPolicyQuickSetup

setmetatable(DiagonalGaussianPolicyQuickSetup, ReinforcementLearningBaseQuickSetup)

function DiagonalGaussianPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDiagonalGaussianPolicyQuickSetup = ReinforcementLearningBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewDiagonalGaussianPolicyQuickSetup, DiagonalGaussianPolicyQuickSetup)
	
	NewDiagonalGaussianPolicyQuickSetup:setName("DiagonalGaussianPolicyQuickSetup")
	
	local actionStandardDeviationVector = parameterDictionary.actionStandardDeviationVector
	
	NewDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector = actionStandardDeviationVector
	
	NewDiagonalGaussianPolicyQuickSetup.actionNoiseVector = parameterDictionary.actionNoiseVector
	
	NewDiagonalGaussianPolicyQuickSetup.previousActionMeanVector = parameterDictionary.previousActionMeanVector

	NewDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector = parameterDictionary.previousActionNoiseVector
	
	NewDiagonalGaussianPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue)
		
		local Model = NewDiagonalGaussianPolicyQuickSetup.Model

		if (not Model) then error("No model!") end
		
		local numberOfReinforcementsPerEpisode = NewDiagonalGaussianPolicyQuickSetup.numberOfReinforcementsPerEpisode

		local currentNumberOfReinforcements = NewDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements

		local currentNumberOfEpisodes = NewDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes
		
		local ExperienceReplay = NewDiagonalGaussianPolicyQuickSetup.ExperienceReplay

		local previousFeatureVector = NewDiagonalGaussianPolicyQuickSetup.previousFeatureVector

		local actionMeanVector = Model:predict(currentFeatureVector, true)
		
		local actionStandardDeviationVector = NewDiagonalGaussianPolicyQuickSetup.actionStandardDeviationVector
		
		local previousActionMeanVector =  NewDiagonalGaussianPolicyQuickSetup.previousActionMeanVector

		local previousActionNoiseVector = NewDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector
		
		local actionNoiseVector = NewDiagonalGaussianPolicyQuickSetup.actionNoiseVector
		
		local terminalStateValue = 0

		if (not actionNoiseVector) then 
			
			local actionVectorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(actionMeanVector)
			
			actionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor(actionVectorDimensionSizeArray) 
			
		end
		
		local actionVector = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)
		
		actionNoiseVector = AqwamTensorLibrary:add(actionNoiseVector, actionMeanVector)
		
		local temporalDifferenceError
		
		if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then terminalStateValue = 1 end

		if (previousFeatureVector) then
			
			local updateFunction = NewDiagonalGaussianPolicyQuickSetup.updateFunction

			currentNumberOfReinforcements = currentNumberOfReinforcements + 1

			temporalDifferenceError = Model:diagonalGaussianUpdate(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction() end

		end

		if (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode) then

			local episodeUpdateFunction = NewDiagonalGaussianPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction() end

		end
		
		if (ExperienceReplay) and (previousFeatureVector) then

			ExperienceReplay:addExperience(previousFeatureVector, previousActionMeanVector, actionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storeTerminalStateValue)

				return Model:diagonalGaussianUpdate(storedPreviousFeatureVector, storedActionMeanVector, storedActionStandardDeviationVector, storedActionNoiseVector, storedRewardValue, storedCurrentFeatureVector, storeTerminalStateValue)

			end)

		end
		
		NewDiagonalGaussianPolicyQuickSetup.totalNumberOfReinforcements = NewDiagonalGaussianPolicyQuickSetup.totalNumberOfReinforcements + 1

		NewDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements = currentNumberOfReinforcements

		NewDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes

		NewDiagonalGaussianPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewDiagonalGaussianPolicyQuickSetup.previousActionMeanVector = actionMeanVector
		
		NewDiagonalGaussianPolicyQuickSetup.previousActionNoiseVector = actionNoiseVector
		
		if (NewDiagonalGaussianPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end
		
		return actionVector
		
	end)
	
	return NewDiagonalGaussianPolicyQuickSetup
	
end

return DiagonalGaussianPolicyQuickSetup
