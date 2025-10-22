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

local CategoricalPolicyBaseQuickSetup = require(script.Parent.CategoricalPolicyBaseQuickSetup)

SingleCategoricalPolicyQuickSetup = {}

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
	
	NewSingleCategoricalPolicyQuickSetup.ExperienceReplay = parameterDictionary.ExperienceReplay
	
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

		local actionVector = Model:predict(currentFeatureVector, true)
		
		local isEpisodeEnd = (currentNumberOfReinforcements >= numberOfReinforcementsPerEpisode)
		
		local terminalStateValue = 0

		local temporalDifferenceError
		
		if (isOriginalValueNotAVector) then currentFeatureVector = currentFeatureVector[1][1] end
		
		local actionIndex, selectedActionCountVector, currentEpsilon = NewSingleCategoricalPolicyQuickSetup:selectAction(actionVector, NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector, NewSingleCategoricalPolicyQuickSetup.currentEpsilon, NewSingleCategoricalPolicyQuickSetup.EpsilonValueScheduler, currentNumberOfReinforcements)

		local action = ActionsList[actionIndex]

		local actionValue = actionVector[1][actionIndex]
		
		if (isEpisodeEnd) then terminalStateValue = 1 end

		if (previousFeatureVector) then
			
			local updateFunction = NewSingleCategoricalPolicyQuickSetup.updateFunction

			temporalDifferenceError = Model:categoricalUpdate(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

			if (updateFunction) then updateFunction(terminalStateValue) end

		end

		if (isEpisodeEnd) then

			local episodeUpdateFunction = NewSingleCategoricalPolicyQuickSetup.episodeUpdateFunction

			currentNumberOfReinforcements = 0

			currentNumberOfEpisodes = currentNumberOfEpisodes + 1

			Model:episodeUpdate(terminalStateValue)

			if episodeUpdateFunction then episodeUpdateFunction(terminalStateValue) end

		end
		
		if (ExperienceReplay) and (previousFeatureVector) then
			
			ExperienceReplay:addExperience(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, terminalStateValue)

			ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

			ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

				return Model:categoricalUpdate(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, storedTerminalStateValue)

			end)
			
		end

		NewSingleCategoricalPolicyQuickSetup.previousFeatureVector = currentFeatureVector
		
		NewSingleCategoricalPolicyQuickSetup.previousAction = action
		
		NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector = selectedActionCountVector
		
		NewSingleCategoricalPolicyQuickSetup.currentEpsilon = currentEpsilon
		
		NewSingleCategoricalPolicyQuickSetup.currentNumberOfReinforcements = currentNumberOfReinforcements

		NewSingleCategoricalPolicyQuickSetup.currentNumberOfEpisodes = currentNumberOfEpisodes
		
		if (NewSingleCategoricalPolicyQuickSetup.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end

		if (returnOriginalOutput) then return actionVector end

		return action, actionValue
		
	end)
	
	NewSingleCategoricalPolicyQuickSetup:setResetFunction(function()

		NewSingleCategoricalPolicyQuickSetup.currentNumberOfReinforcements = 0

		NewSingleCategoricalPolicyQuickSetup.currentNumberOfEpisodes = 1

		NewSingleCategoricalPolicyQuickSetup.previousFeatureVector = nil

		NewSingleCategoricalPolicyQuickSetup.previousAction = nil

		NewSingleCategoricalPolicyQuickSetup.selectedActionCountVector = nil

	end)
	
	return NewSingleCategoricalPolicyQuickSetup
	
end

return SingleCategoricalPolicyQuickSetup
