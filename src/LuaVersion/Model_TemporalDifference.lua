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

TemporalDifferenceModel = {}

TemporalDifferenceModel.__index = TemporalDifferenceModel

setmetatable(TemporalDifferenceModel, DeepReinforcementLearningBaseModel)

function TemporalDifferenceModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewTemporalDifferenceModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewTemporalDifferenceModel, TemporalDifferenceModel)
	
	NewTemporalDifferenceModel:setName("TemporalDifference")
	
	NewTemporalDifferenceModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local Model = NewTemporalDifferenceModel.Model
		
		local discountFactor = NewTemporalDifferenceModel.discountFactor

		local currentQVector = Model:forwardPropagate(currentFeatureVector)

		local previousQVector = Model:forwardPropagate(previousFeatureVector)
		
		local targetValue = rewardValue + (discountFactor * currentQVector[1][1] * (1 - terminalStateValue))
		
		local temporalDifferenceError = targetValue - previousQVector[1][1]
		
		local negatedTemporalDifferenceErrorVector = {{-temporalDifferenceError}}
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		return temporalDifferenceError

	end)
	
	NewTemporalDifferenceModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
	end)
	
	NewTemporalDifferenceModel:setResetFunction(function() 
		
	end)

	return NewTemporalDifferenceModel

end

function TemporalDifferenceModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return TemporalDifferenceModel
