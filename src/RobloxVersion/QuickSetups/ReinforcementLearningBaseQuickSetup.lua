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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

ReinforcementLearningBaseQuickSetup = {}

ReinforcementLearningBaseQuickSetup.__index = ReinforcementLearningBaseQuickSetup

setmetatable(ReinforcementLearningBaseQuickSetup, BaseInstance)

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultIsOutputPrinted = true

function ReinforcementLearningBaseQuickSetup.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewReinforcementLearningBaseQuickSetup = BaseInstance.new(parameterDictionary)

	setmetatable(NewReinforcementLearningBaseQuickSetup, ReinforcementLearningBaseQuickSetup)
	
	NewReinforcementLearningBaseQuickSetup:setName("ReinforcementLearningBaseQuickSetup")

	NewReinforcementLearningBaseQuickSetup:setClassName("ReinforcementLearningQuickSetup")

	NewReinforcementLearningBaseQuickSetup.isOutputPrinted = NewReinforcementLearningBaseQuickSetup:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, defaultIsOutputPrinted)  

	NewReinforcementLearningBaseQuickSetup.numberOfReinforcementsPerEpisode = parameterDictionary.numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewReinforcementLearningBaseQuickSetup.Model = parameterDictionary.Model
	
	NewReinforcementLearningBaseQuickSetup.reinforceFunction = parameterDictionary.reinforceFunction
	
	NewReinforcementLearningBaseQuickSetup.updateFunction = parameterDictionary.updateFunction

	NewReinforcementLearningBaseQuickSetup.episodeUpdateFunction = parameterDictionary.episodeUpdateFunction
	
	NewReinforcementLearningBaseQuickSetup.resetFunction = parameterDictionary.resetFunction

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

function ReinforcementLearningBaseQuickSetup:setResettFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function ReinforcementLearningBaseQuickSetup:reinforce(...)
	
	return self.reinforceFunction(...)

end

function ReinforcementLearningBaseQuickSetup:reset(...)

	return self.reset(...)

end

function ReinforcementLearningBaseQuickSetup:setModel(Model)

	self.Model = Model

end

function ReinforcementLearningBaseQuickSetup:getModel()

	return self.Model

end

return ReinforcementLearningBaseQuickSetup
