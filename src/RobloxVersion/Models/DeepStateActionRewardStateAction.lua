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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepStateActionRewardStateActionModel = {}

DeepStateActionRewardStateActionModel.__index = DeepStateActionRewardStateActionModel

setmetatable(DeepStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

function DeepStateActionRewardStateActionModel.new(discountFactor)

	local NewDeepStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepStateActionRewardStateActionModel, DeepStateActionRewardStateActionModel)

	NewDeepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepStateActionRewardStateActionModel.Model

		local qVector = Model:predict(currentFeatureVector, true)

		local discountedQVector = AqwamMatrixLibrary:multiply(NewDeepStateActionRewardStateActionModel.discountFactor, qVector)

		local targetVector = AqwamMatrixLibrary:add(rewardValue, discountedQVector)

		local previousQVector = Model:predict(previousFeatureVector, true)

		local temporalDifferenceVector = AqwamMatrixLibrary:subtract(targetVector, previousQVector)
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:backwardPropagate(temporalDifferenceVector, true)
		
		return temporalDifferenceVector

	end)
	
	NewDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function() end)
	
	NewDeepStateActionRewardStateActionModel:setResetFunction(function() end)

	return NewDeepStateActionRewardStateActionModel

end

function DeepStateActionRewardStateActionModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepStateActionRewardStateActionModel