--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

ReinforcementLearningActorCriticBaseModel = {}

ReinforcementLearningActorCriticBaseModel.__index = ReinforcementLearningActorCriticBaseModel

setmetatable(ReinforcementLearningActorCriticBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewReinforcementLearningActorCriticBaseModel = {}
	
	setmetatable(NewReinforcementLearningActorCriticBaseModel, ReinforcementLearningActorCriticBaseModel)
	
	NewReinforcementLearningActorCriticBaseModel:setName("ReinforcementLearningActorCriticBaseModel")

	NewReinforcementLearningActorCriticBaseModel:setClassName("ReinforcementLearningActorCriticModel")

	NewReinforcementLearningActorCriticBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor

	NewReinforcementLearningActorCriticBaseModel.ActorModel = parameterDictionary.ActorModel
	
	NewReinforcementLearningActorCriticBaseModel.CriticModel = parameterDictionary.CriticModel
	
	return NewReinforcementLearningActorCriticBaseModel
	
end

function ReinforcementLearningActorCriticBaseModel:setDiscountFactor(discountFactor)

	self.discountFactor = discountFactor

end

function ReinforcementLearningActorCriticBaseModel:getDiscountFactor()

	return self.discountFactor

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

function ReinforcementLearningActorCriticBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningActorCriticBaseModel:getClassesList()

	return self.ActorModel:getClassesList()

end

function ReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The categorical update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then
		
		if (not actionStandardDeviationVector) then error("No action standard deviation vector.") end

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction

	if (episodeUpdateFunction) then

		return episodeUpdateFunction(terminalStateValue)

	else

		error("The episode update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningActorCriticBaseModel:reset()

	local resetFunction = self.resetFunction

	if (resetFunction) then 

		return resetFunction() 

	else

		error("The reset function is not implemented!")

	end

end

return ReinforcementLearningActorCriticBaseModel