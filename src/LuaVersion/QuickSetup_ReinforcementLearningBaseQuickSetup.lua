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

local BaseInstance = require("Core_BaseInstance")

ReinforcementLearningBaseQuickSetup = {}

ReinforcementLearningBaseQuickSetup.__index = ReinforcementLearningBaseQuickSetup

setmetatable(ReinforcementLearningBaseQuickSetup, BaseInstance)

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0

local defaultIsOutputPrinted = true

local defaultTotalNumberOfReinforcements = 0

local defaultCurrentNumberOfReinforcements = 0

local defaultCurrentNumberOfEpisodes = 0

function ReinforcementLearningBaseQuickSetup.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewReinforcementLearningBaseQuickSetup = BaseInstance.new(parameterDictionary)

	setmetatable(NewReinforcementLearningBaseQuickSetup, ReinforcementLearningBaseQuickSetup)
	
	NewReinforcementLearningBaseQuickSetup:setName("ReinforcementLearningBaseQuickSetup")

	NewReinforcementLearningBaseQuickSetup:setClassName("ReinforcementLearningQuickSetup")

	NewReinforcementLearningBaseQuickSetup.isOutputPrinted = NewReinforcementLearningBaseQuickSetup:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, defaultIsOutputPrinted)  

	NewReinforcementLearningBaseQuickSetup.numberOfReinforcementsPerEpisode = parameterDictionary.numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewReinforcementLearningBaseQuickSetup.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewReinforcementLearningBaseQuickSetup.currentEpsilon = parameterDictionary.currentEpsilon or parameterDictionary.epsilon or defaultEpsilon

	NewReinforcementLearningBaseQuickSetup.Model = parameterDictionary.Model

	NewReinforcementLearningBaseQuickSetup.ExperienceReplay = parameterDictionary.ExperienceReplay

	NewReinforcementLearningBaseQuickSetup.EpsilonValueScheduler = parameterDictionary.EpsilonValueScheduler
	
	NewReinforcementLearningBaseQuickSetup.totalNumberOfReinforcements = parameterDictionary.totalNumberOfReinforcements or defaultTotalNumberOfReinforcements

	NewReinforcementLearningBaseQuickSetup.currentNumberOfReinforcements = parameterDictionary.currentNumberOfReinforcements or defaultCurrentNumberOfReinforcements

	NewReinforcementLearningBaseQuickSetup.currentNumberOfEpisodes = parameterDictionary.currentNumberOfEpisodes or defaultCurrentNumberOfEpisodes
	
	NewReinforcementLearningBaseQuickSetup.reinforceFunction = parameterDictionary.reinforceFunction
	
	NewReinforcementLearningBaseQuickSetup.updateFunction = parameterDictionary.updateFunction

	NewReinforcementLearningBaseQuickSetup.episodeUpdateFunction =  parameterDictionary.episodeUpdateFunction
	
	NewReinforcementLearningBaseQuickSetup.previousFeatureVector = parameterDictionary.previousFeatureVector

	return NewReinforcementLearningBaseQuickSetup

end

function ReinforcementLearningBaseQuickSetup:setReinforceFunction(reinforceFunction)
	
	self.reinforceFunction = reinforceFunction
	
end

function ReinforcementLearningBaseQuickSetup:extendUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningBaseQuickSetup:extendEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningBaseQuickSetup:setPrintOutput(option)

	self.isOutputPrinted = self:getValueOrDefaultValue(option, self.isOutputPrinted)

end

function ReinforcementLearningBaseQuickSetup:reinforce(...)
	
	return self.reinforceFunction(...)

end

function ReinforcementLearningBaseQuickSetup:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

function ReinforcementLearningBaseQuickSetup:setModel(Model)

	self.Model = Model

end

function ReinforcementLearningBaseQuickSetup:setEpsilonValueScheduler(EpsilonValueScheduler)

	self.EpsilonValueScheduler = EpsilonValueScheduler

end

function ReinforcementLearningBaseQuickSetup:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function ReinforcementLearningBaseQuickSetup:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function ReinforcementLearningBaseQuickSetup:getCurrentEpsilon()

	return self.currentEpsilon

end

function ReinforcementLearningBaseQuickSetup:getModel()

	return self.Model

end

function ReinforcementLearningBaseQuickSetup:getExperienceReplay()

	return self.ExperienceReplay

end

function ReinforcementLearningBaseQuickSetup:getEpsilonValueScheduler()

	return self.EpsilonValueScheduler

end

function ReinforcementLearningBaseQuickSetup:reset()

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	local Model = self.Model

	local ExperienceReplay = self.ExperienceReplay

	if (Model) then Model:reset() end

	if (ExperienceReplay) then ExperienceReplay:reset() end

end

return ReinforcementLearningBaseQuickSetup
