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

local BaseInstance = require("Core_BaseInstance")

ReinforcementLearningBaseModel = {}

ReinforcementLearningBaseModel.__index = ReinforcementLearningBaseModel

setmetatable(ReinforcementLearningBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function ReinforcementLearningBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewReinforcementLearningBaseModel = {}
	
	setmetatable(NewReinforcementLearningBaseModel, ReinforcementLearningBaseModel)
	
	NewReinforcementLearningBaseModel:setName("ReinforcementLearningBaseModel")

	NewReinforcementLearningBaseModel:setClassName("ReinforcementLearningModel")
	
	NewReinforcementLearningBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor
	
	NewReinforcementLearningBaseModel.Model = parameterDictionary.Model
	
	return NewReinforcementLearningBaseModel
	
end

function ReinforcementLearningBaseModel:setDiscountFactor(discountFactor)
	
	self.discountFactor = discountFactor
	
end

function ReinforcementLearningBaseModel:getDiscountFactor()
	
	return self.discountFactor
	
end

function ReinforcementLearningBaseModel:setModel(Model)
	
	self.Model = Model
	
end

function ReinforcementLearningBaseModel:getModel()

	return self.Model

end

function ReinforcementLearningBaseModel:predict(featureVector, returnOriginalOutput)

	return self.Model:predict(featureVector, returnOriginalOutput)

end

function ReinforcementLearningBaseModel:getClassesList()
	
	return self.Model:getClassesList()
	
end

function ReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function ReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)
	
	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction
	
end

function ReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
	
	local categoricalUpdateFunction = self.categoricalUpdateFunction
	
	if (categoricalUpdateFunction) then
		
		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
	else
		
		error("The categorical update function is not implemented.")
		
	end

end

function ReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then
		
		if (not actionStandardDeviationVector) then error("No action standard deviation vector.") end

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented.")

	end

end

function ReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if (episodeUpdateFunction) then
		
		return episodeUpdateFunction(terminalStateValue)
		
	else
		
		error("The episode update function is not implemented.")
		
	end

end

function ReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningBaseModel:reset()
	
	local resetFunction = self.resetFunction

	if (resetFunction) then 
		
		return resetFunction() 
		
	else
		
		error("The reset function is not implemented.")
		
	end

end

return ReinforcementLearningBaseModel