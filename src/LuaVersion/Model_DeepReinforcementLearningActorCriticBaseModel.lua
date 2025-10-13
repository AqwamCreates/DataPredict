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

DeepReinforcementLearningActorCriticBaseModel = {}

DeepReinforcementLearningActorCriticBaseModel.__index = DeepReinforcementLearningActorCriticBaseModel

setmetatable(DeepReinforcementLearningActorCriticBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDeepReinforcementLearningActorCriticBaseModel = {}
	
	setmetatable(NewDeepReinforcementLearningActorCriticBaseModel, DeepReinforcementLearningActorCriticBaseModel)
	
	NewDeepReinforcementLearningActorCriticBaseModel:setName("DeepReinforcementLearningActorCriticBaseModel")

	NewDeepReinforcementLearningActorCriticBaseModel:setClassName("DeepReinforcementLearningActorCriticModel")

	NewDeepReinforcementLearningActorCriticBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewDeepReinforcementLearningActorCriticBaseModel.ActorModel = parameterDictionary.ActorModel
	
	NewDeepReinforcementLearningActorCriticBaseModel.CriticModel = parameterDictionary.CriticModel
	
	return NewDeepReinforcementLearningActorCriticBaseModel
	
end

function DeepReinforcementLearningActorCriticBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function DeepReinforcementLearningActorCriticBaseModel:getDiscountFactor()

	return self.discountFactor

end

function DeepReinforcementLearningActorCriticBaseModel:setActorModel(ActorModel)
	
	self.ActorModel = ActorModel
	
end

function DeepReinforcementLearningActorCriticBaseModel:setCriticModel(CriticModel)

	self.CriticModel = CriticModel
	
end

function DeepReinforcementLearningActorCriticBaseModel:getActorModel()

	return self.ActorModel

end

function DeepReinforcementLearningActorCriticBaseModel:getCriticModel()

	return self.CriticModel

end

function DeepReinforcementLearningActorCriticBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function DeepReinforcementLearningActorCriticBaseModel:getActionsList()

	return self.ActorModel:getClassesList()

end

function DeepReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function DeepReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function DeepReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	return self.categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

end

function DeepReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	return self.diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

end

function DeepReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function DeepReinforcementLearningActorCriticBaseModel:episodeUpdate(terminalStateValue)

	return self.episodeUpdateFunction(terminalStateValue)

end

function DeepReinforcementLearningActorCriticBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function DeepReinforcementLearningActorCriticBaseModel:reset()

	return self.resetFunction() 

end

return DeepReinforcementLearningActorCriticBaseModel
