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

function DeepReinforcementLearningBaseModel:setModelParameters(ModelParameters, doNotDeepCopy)

	self.Model:setModelParameters(ModelParameters, doNotDeepCopy)

end

function DeepReinforcementLearningBaseModel:getModelParameters(doNotDeepCopy)

	return self.Model:getModelParameters(doNotDeepCopy)

end

function DeepReinforcementLearningBaseModel:predict(featureVector, returnOriginalOutput)

	return self.Model:predict(featureVector, returnOriginalOutput)

end

function DeepReinforcementLearningBaseModel:getActionsList()
	
	return self.Model:getClassesList()
	
end

function DeepReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function DeepReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)
	
	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction
	
end

function DeepReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
	
	return self.categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

end

function DeepReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

	return self.diagonalGaussianUpdateFunction(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

end

function DeepReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function DeepReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)

	return self.episodeUpdateFunction(terminalStateValue)

end

function DeepReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function DeepReinforcementLearningBaseModel:reset()
	
	self.resetFunction() 

end

return DeepReinforcementLearningBaseModel
