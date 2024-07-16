local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningDeepDuelingQLearningBaseModel = require(script.Parent.ReinforcementLearningDeepDuelingQLearningBaseModel)

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

		ValueModel:backPropagate(vLoss, true)

		return vLoss
		
	end)

	return NewDeepDuelingQLearning

end

return DeepDuelingQLearning
