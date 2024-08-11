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

local ReinforcementLearningDeepDuelingQLearningBaseModel = require("Model_ReinforcementLearningDeepDuelingQLearningBaseModel")

DeepDoubleDuelingQLearning = {}

DeepDoubleDuelingQLearning.__index = DeepDoubleDuelingQLearning

setmetatable(DeepDoubleDuelingQLearning, ReinforcementLearningDeepDuelingQLearningBaseModel)

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

function DeepDoubleDuelingQLearning.new(averagingRate, discountFactor)

	local NewDeepDuelingQLearning = ReinforcementLearningDeepDuelingQLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepDuelingQLearning, DeepDoubleDuelingQLearning)
	
	NewDeepDuelingQLearning.averagingRate = averagingRate or defaultAveragingRate
	
	NewDeepDuelingQLearning:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local AdvantageModel = NewDeepDuelingQLearning.AdvantageModel

		local ValueModel = NewDeepDuelingQLearning.ValueModel

		local averagingRate = NewDeepDuelingQLearning.averagingRate

		if (AdvantageModel:getModelParameters() == nil) then AdvantageModel:generateLayers() end

		if (ValueModel:getModelParameters() == nil) then ValueModel:generateLayers() end

		local AdvantageModelPrimaryModelParameters = AdvantageModel:getModelParameters(true)

		local ValueModelPrimaryModelParameters = ValueModel:getModelParameters(true)

		local qLossVector, vLoss = NewDeepDuelingQLearning:generateLoss(previousFeatureVector, action, rewardValue, currentFeatureVector)

		AdvantageModel:forwardPropagate(previousFeatureVector, true)

		AdvantageModel:backwardPropagate(qLossVector, true)

		ValueModel:forwardPropagate(previousFeatureVector, true)

		ValueModel:backwardPropagate(vLoss, true)

		local AdvantageModelTargetModelParameters = AdvantageModel:getModelParameters(true)

		local ValueModelTargetModelParameters = ValueModel:getModelParameters(true)

		AdvantageModelTargetModelParameters = rateAverageModelParameters(averagingRate, AdvantageModelTargetModelParameters, AdvantageModelPrimaryModelParameters)

		ValueModelTargetModelParameters = rateAverageModelParameters(averagingRate, ValueModelTargetModelParameters, ValueModelPrimaryModelParameters)

		AdvantageModel:setModelParameters(AdvantageModelTargetModelParameters, true)

		ValueModel:setModelParameters(ValueModelTargetModelParameters, true)

		return vLoss
		
	end)

	return NewDeepDuelingQLearning

end

function DeepDoubleDuelingQLearning:setParameters(averagingRate, discountFactor)
	
	self.averagingRate = averagingRate or self.averagingRate

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepDoubleDuelingQLearning
