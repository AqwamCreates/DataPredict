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

local ReinforcementLearningDeepDuelingQLearningBaseModel = require("Model_ReinforcementLearningDeepDuelingQLearningBaseModel")

DeepDuelingQLearning = {}

DeepDuelingQLearning.__index = DeepDuelingQLearning

setmetatable(DeepDuelingQLearning, ReinforcementLearningDeepDuelingQLearningBaseModel)

function DeepDuelingQLearning.new(discountFactor)

	local NewDeepDuelingQLearning = ReinforcementLearningDeepDuelingQLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepDuelingQLearning, DeepDuelingQLearning)
	
	NewDeepDuelingQLearning:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local AdvantageModel = NewDeepDuelingQLearning.AdvantageModel

		local ValueModel = NewDeepDuelingQLearning.ValueModel

		local qLossVector, vLoss = NewDeepDuelingQLearning:generateLoss(previousFeatureVector, action, rewardValue, currentFeatureVector)

		AdvantageModel:forwardPropagate(previousFeatureVector, true)

		AdvantageModel:backPropagate(qLossVector, true)

		ValueModel:forwardPropagate(previousFeatureVector, true)

		ValueModel:backwardPropagate(vLoss, true)

		return vLoss
		
	end)

	return NewDeepDuelingQLearning

end

return DeepDuelingQLearning
