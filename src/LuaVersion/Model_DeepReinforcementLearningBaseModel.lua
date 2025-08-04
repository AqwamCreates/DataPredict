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

local BaseInstance = require("Cores_BaseInstance")

DeepReinforcementLearningBaseModel = {}

DeepReinforcementLearningBaseModel.__index = DeepReinforcementLearningBaseModel

setmetatable(DeepReinforcementLearningBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function DeepReinforcementLearningBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDeepReinforcementLearningBaseModel = {}
	
	setmetatable(NewDeepReinforcementLearningBaseModel, DeepReinforcementLearningBaseModel)
	
	NewDeepReinforcementLearningBaseModel:setName("DeepReinforcementLearningBaseModel")

	NewDeepReinforcementLearningBaseModel:setClassName("DeepReinforcementLearningModel")
	
	NewDeepReinforcementLearningBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor
	
	NewDeepReinforcementLearningBaseModel.Model = parameterDictionary.Model
	
	return NewDeepReinforcementLearningBaseModel
	
end

function DeepReinforcementLearningBaseModel:setDiscountFactor(discountFactor)
	
	self.discountFactor = discountFactor
	
end

function DeepReinforcementLearningBaseModel:getDiscountFactor()
	
	return self.discountFactor
	
end

function DeepReinforcementLearningBaseModel:setModel(Model)
	
	self.Model = Model
	
end

function DeepReinforcementLearningBaseModel:getModel()

	return self.Model

end

function DeepReinforcementLearningBaseModel:predict(featureVector, returnOriginalOutput)

	return self.Model:predict(featureVector, returnOriginalOutput)

end

function DeepReinforcementLearningBaseModel:getClassesList()
	
	return self.Model:getClassesList()
	
end

function DeepReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function DeepReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)
	
	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction
	
end

function DeepReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
	
	local categoricalUpdateFunction = self.categoricalUpdateFunction
	
	if (categoricalUpdateFunction) then
		
		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
	else
		
		error("The categorical update function is not implemented.")
		
	end

end

function DeepReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction

	if (diagonalGaussianUpdateFunction) then
		
		if (not actionStandardDeviationVector) then error("No action standard deviation vector.") end

		return diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	else

		error("The diagonal Gaussian update function is not implemented.")

	end

end

function DeepReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function DeepReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)

	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if (episodeUpdateFunction) then
		
		return episodeUpdateFunction(terminalStateValue)
		
	else
		
		error("The episode update function is not implemented.")
		
	end

end

function DeepReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function DeepReinforcementLearningBaseModel:reset()
	
	local resetFunction = self.resetFunction

	if (resetFunction) then 
		
		return resetFunction() 
		
	else
		
		error("The reset function is not implemented.")
		
	end

end

return DeepReinforcementLearningBaseModel
