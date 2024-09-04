--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

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

function ReinforcementLearningActorCriticBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

	else

		error("The categorical update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

	else

		error("The diagonal Gaussian update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:episodeUpdate()

	local episodeUpdateFunction = self.episodeUpdateFunction

	if (episodeUpdateFunction) then

		return episodeUpdateFunction()

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

function ReinforcementLearningActorCriticBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningActorCriticBaseModel