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

function ReinforcementLearningActorCriticBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:setCategoricalEpisodeUpdateFunction(categoricalEpisodeUpdateFunction)

	self.categoricalEpisodeUpdateFunction = categoricalEpisodeUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)

	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:setDiagonalGaussianEpisodeUpdateFunction(diagonalGaussianEpisodeUpdateFunction)

	self.diagonalGaussianEpisodeUpdateFunction = diagonalGaussianEpisodeUpdateFunction

end

function ReinforcementLearningActorCriticBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningActorCriticBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local categoricalUpdateFunction = self.categoricalUpdateFunction

	if (categoricalUpdateFunction) then

		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

	else

		error("Categorical update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:categoricalEpisodeUpdate()

	local categoricalEpisodeUpdateFunction = self.categoricalEpisodeUpdateFunction

	if (categoricalEpisodeUpdateFunction) then

		return categoricalEpisodeUpdateFunction()

	else

		error("Categorical episode update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

	else

		error("Diagonal Gaussian update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:diagonalGaussianEpisodeUpdate()

	local diagonalGaussianEpisodeUpdateFunction = self.diagonalGaussianEpisodeUpdateFunction

	if (diagonalGaussianEpisodeUpdateFunction) then

		return diagonalGaussianEpisodeUpdateFunction()

	else

		error("Diagonal Gaussian episode update function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:setCategoricalResetFunction(categoricalResetFunction)

	self.categoricalResetFunction = categoricalResetFunction

end

function ReinforcementLearningActorCriticBaseModel:setDiagonalGaussianResetFunction(diagonalGaussianResetFunction)

	self.diagonalGaussianResetFunction = diagonalGaussianResetFunction

end

function ReinforcementLearningActorCriticBaseModel:categoricalReset()

	local categoricalResetFunction = self.categoricalResetFunction

	if (categoricalResetFunction) then 

		return categoricalResetFunction() 

	else

		error("Categorical reset function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:diagonalGaussianReset()

	local diagonalGaussianResetFunction = self.diagonalGaussianResetFunction

	if (diagonalGaussianResetFunction) then 

		return diagonalGaussianResetFunction()

	else

		error("Diagonal Gaussian reset function is not implemented!")

	end

end

function ReinforcementLearningActorCriticBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningActorCriticBaseModel