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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local DeepReinforcementLearningBaseModel = require("Model_DeepReinforcementLearningBaseModel")

DeepTemporalDifferenceModel = {}

DeepTemporalDifferenceModel.__index = DeepTemporalDifferenceModel

setmetatable(DeepTemporalDifferenceModel, DeepReinforcementLearningBaseModel)

function DeepTemporalDifferenceModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepTemporalDifferenceModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepTemporalDifferenceModel, DeepTemporalDifferenceModel)
	
	NewDeepTemporalDifferenceModel:setName("DeepTemporalDifference")
	
	NewDeepTemporalDifferenceModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local Model = NewDeepTemporalDifferenceModel.Model
		
		local discountFactor = NewDeepTemporalDifferenceModel.discountFactor

		local currentQVector = Model:forwardPropagate(currentFeatureVector)

		local previousQVector = Model:forwardPropagate(previousFeatureVector)
		
		local targetValue = rewardValue + (discountFactor * currentQVector[1][1] * (1 - terminalStateValue))
		
		local DeepTemporalDifferenceError = targetValue - previousQVector[1][1]
		
		local negatedDeepTemporalDifferenceErrorVector = {{-DeepTemporalDifferenceError}}
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedDeepTemporalDifferenceErrorVector, true)
		
		return DeepTemporalDifferenceError

	end)
	
	NewDeepTemporalDifferenceModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
	end)
	
	NewDeepTemporalDifferenceModel:setResetFunction(function() 
		
	end)

	return NewDeepTemporalDifferenceModel

end

function DeepTemporalDifferenceModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return DeepTemporalDifferenceModel
