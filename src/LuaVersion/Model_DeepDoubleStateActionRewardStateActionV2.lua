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

local ReinforcementLearningBaseModel = require("BaseModel_ReinforcementLearningBaseModel")

DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamMatrixLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamMatrixLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamMatrixLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDoubleStateActionRewardStateActionModel.new(averagingRate, discountFactor)

	local NewDeepDoubleStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)

	NewDeepDoubleStateActionRewardStateActionModel.averagingRate = averagingRate or defaultAveragingRate

	NewDeepDoubleStateActionRewardStateActionModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model
		
		local PrimaryModelParameters = Model:getModelParameters(true)

		if (PrimaryModelParameters == nil) then 
			
			Model:generateLayers() 
			PrimaryModelParameters = Model:getModelParameters(true)
			
		end
		
		local qVector = Model:predict(currentFeatureVector, true)

		local discountedQVector = AqwamMatrixLibrary:multiply(NewDeepDoubleStateActionRewardStateActionModel.discountFactor, qVector)

		local targetVector = AqwamMatrixLibrary:add(rewardValue, discountedQVector)

		local previousQVector = Model:predict(previousFeatureVector, true)

		local temporalDifferenceVector = AqwamMatrixLibrary:subtract(targetVector, previousQVector)

		Model:forwardPropagate(previousFeatureVector, true)

		Model:backPropagate(temporalDifferenceVector, true)
		
		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceVector

	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

function DeepDoubleStateActionRewardStateActionModel:setParameters(averagingRate, discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

end

return DeepDoubleStateActionRewardStateActionModel
