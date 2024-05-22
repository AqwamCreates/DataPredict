local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

ReinforcementLearningActorCriticBaseModel = {}

ReinforcementLearningActorCriticBaseModel.__index = ReinforcementLearningActorCriticBaseModel

local defaultDiscountFactor = 0.95

function ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	local NewReinforcementLearningActorCriticBaseModel = {}
	
	setmetatable(NewReinforcementLearningActorCriticBaseModel, ReinforcementLearningActorCriticBaseModel)
	
	NewReinforcementLearningActorCriticBaseModel.discountFactor = discountFactor or defaultDiscountFactor
	
	return NewReinforcementLearningActorCriticBaseModel
	
end

function ReinforcementLearningActorCriticBaseModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor
	
end

function ReinforcementLearningActorCriticBaseModel:setActorModel(ActorModel)
	
	self.ActorModel = ActorModel
	
end

function ReinforcementLearningActorCriticBaseModel:setCriticModel(CriticModel)

	self.CriticModel = CriticModel
	
end

function ReinforcementLearningActorCriticBaseModel:getActorModel()

	return self.ActorModel

end

function ReinforcementLearningActorCriticBaseModel:getCriticModel()

	return self.CriticModel

end

function ReinforcementLearningActorCriticBaseModel:setUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningActorCriticBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

end

function ReinforcementLearningActorCriticBaseModel:episodeUpdate()

	local episodeUpdateFunction = self.episodeUpdateFunction

	if not episodeUpdateFunction then return end

	episodeUpdateFunction()

end

function ReinforcementLearningActorCriticBaseModel:extendResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningActorCriticBaseModel:reset()

	if (self.resetFunction) then self.resetFunction() end

end

function ReinforcementLearningActorCriticBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningActorCriticBaseModel
