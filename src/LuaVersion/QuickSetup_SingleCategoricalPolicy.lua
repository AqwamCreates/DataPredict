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

local SingleCategoricalPolicyQuickSetup = {}

SingleCategoricalPolicyQuickSetup.__index = SingleCategoricalPolicyQuickSetup

setmetatable(SingleCategoricalPolicyQuickSetup, CategoricalPolicyBaseQuickSetup)

local defaultCurrentNumberOfReinforcements = 0

local defaultCurrentNumberOfEpisodes = 1

function SingleCategoricalPolicyQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSingleCategoricalPolicyQuickSetup = CategoricalPolicyBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewSingleCategoricalPolicyQuickSetup, SingleCategoricalPolicyQuickSetup)
	
	NewSingleCategoricalPolicyQuickSetup:setName("SingleCategoricalPolicyQuickSetup")
	
	NewSingleCategoricalPolicyQuickSetup.previousAction = parameterDictionary.previousAction
	
	NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector = parameterDictionary.selectedActionCountVector
	
	NewSingleCategoricalPolicyQuickSetup.currentNumberOfReinforcements = parameterDictionary.currentNumberOfReinforcements or defaultCurrentNumberOfReinforcements

	NewSingleCategoricalPolicyQuickSetup.currentNumberOfEpisodes = parameterDictionary.currentNumberOfEpisodes or defaultCurrentNumberOfEpisodes
	
	NewSingleCategoricalPolicyQuickSetup:setReinforceFunction(function(currentFeatureVector, rewardValue, returnOriginalOutput)
		
		local Model = NewSingleCategoricalPolicyQuickSetup.Model
		
		if (not Model) then error("No model.") end
		
		local isOriginalValueNotAVector = (type(currentFeatureVector) ~= "table")
		
		if (isOriginalValueNotAVector) then currentFeatureVector = {{currentFeatureVector}} end
		
		local numberOfReinforcementsPerEpisode = NewSingleCategoricalPolicyQuickSetup.numberOfReinforcementsPerEpisode

		local currentNumberOfReinforcements = NewSingleCategoricalPolicyQuickSetup.currentNumberOfReinforcements + 1

		local currentNumberOfEpisodes = NewSingleCategoricalPolicyQuickSetup.currentNumberOfEpisodes
		
		local ExperienceReplay = NewSingleCategoricalPolicyQuickSetup.ExperienceReplay
		
		local previousFeatureVector = NewSingleCategoricalPolicyQuickSetup.previousFeatureVector
		
		local previousAction = NewSingleCategoricalPolicyQuickSetup.previousAction

		local ActionsList = Model:getActionsList()

		local currentActionVector = Model:predict(currentFeatureVector, true)
		
		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)
		
		local terminalStateValue = (isEpisodeEnd and 1) or 0
		
		if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end
		
		local currentActionIndex, selectedActionCountVector, currentEpsilon, currentTemperature, currentCValue = NewSingleCategoricalPolicyQuickSetup:selectAction(currentActionVector, NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector, NewSingleCategoricalPolicyQuickSetup.currentEpsilon, NewSingleCategoricalPolicyQuickSetup.currentTemperature, NewSingleCategoricalPolicyQuickSetup.currentCValue, NewSingleCategoricalPolicyQuickSetup.EpsilonValueScheduler, NewSingleCategoricalPolicyQuickSetup.TemperatureValueScheduler, NewSingleCategoricalPolicyQuickSetup.CValueValueScheduler, currentNumberOfReinforcements)

		local currentAction = ActionsList[currentActionIndex]

		local currentActionValue = currentActionVector[1][currentActionIndex]
		
		local temporalDifferenceError

		if (previousFeatureVector) then
			
			local updateFunction = NewSingleCategoricalPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			local episodeUpdateFunction = NewSingleCategoricalPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if (episodeUpdateFunction) then episodeUpdateFunction(terminalStateValue) end

		end
		
		if (ExperienceReplay) and (previousFeatureVector) then
			
			ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

				return Model:categoricalUpdate(storedPreviousFeatureVector, storedPreviousAction, storedRewardValue, storedCurrentFeatureVector, storedCurrentAction, storedTerminalStateValue)

			end)
			
		end

		NewSingleCategoricalPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewSingleCategoricalPolicyQuickSetup.previousAction = currentAction
		
		NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector = selectedActionCountVector
		
		NewSingleCategoricalPolicyQuickSetup.currentEpsilon = currentEpsilon
		
		NewSingleCategoricalPolicyQuickSetup.currentTemperature = currentTemperature
		
		NewSingleCategoricalPolicyQuickSetup.currentCValue = currentCValue
		
		NewSingleCategoricalPolicyQuickSetup.currentNumberOfReinforcements = currentNumberOfReinforcements

		NewSingleCategoricalPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes
		
		if (NewSingleCategoricalPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end

		if (returnOriginalOutput) then return currentActionVector end

		return currentAction, currentActionValue
		
	end)
	
	NewSingleCategoricalPolicyQuickSetup:setResetFunction(function()

		NewSingleCategoricalPolicyQuickSetup.currentNumberOfReinforcements = 0

		NewSingleCategoricalPolicyQuickSetup.currentNumberOfEpisodes = 1

		NewSingleCategoricalPolicyQuickSetup.previousFeatureVector = nil

		NewSingleCategoricalPolicyQuickSetup.previousAction = nil
		
		NewSingleCategoricalPolicyQuickSetup.currentEpsilon = nil

		NewSingleCategoricalPolicyQuickSetup.currentTemperature = nil
		
		NewSingleCategoricalPolicyQuickSetup.currentCValue = nil

		NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector = nil

	end)
	
	return NewSingleCategoricalPolicyQuickSetup
	
end

return SingleCategoricalPolicyQuickSetup
