local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ReinforcementLearningDeepDuelingQLearning = {}

ReinforcementLearningDeepDuelingQLearning.__index = ReinforcementLearningDeepDuelingQLearning

local defaultDiscountFactor = 0.95

function ReinforcementLearningDeepDuelingQLearning.new(discountFactor)
	
	local NewReinforcementLearningActorCriticBaseModel = {}
	
	setmetatable(NewReinforcementLearningActorCriticBaseModel, ReinforcementLearningDeepDuelingQLearning)
	
	NewReinforcementLearningActorCriticBaseModel.discountFactor = discountFactor or defaultDiscountFactor
	
	return NewReinforcementLearningActorCriticBaseModel
	
end

function ReinforcementLearningDeepDuelingQLearning:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor
	
end

function ReinforcementLearningDeepDuelingQLearning:forwardPropagate(featureVector)

	local vValue = self.ValueModel:predict(featureVector, true)[1][1]

	local advantageMatrix = self.AdvantageModel:predict(featureVector, true)

	local meanAdvantageVector = AqwamMatrixLibrary:horizontalMean(advantageMatrix)

	local qValueVectorPart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValueVector = AqwamMatrixLibrary:add(vValue, qValueVectorPart1)

	return qValueVector, vValue

end

function ReinforcementLearningDeepDuelingQLearning:generateLoss(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local previousQValue, previousVValue = self:forwardPropagate(previousFeatureVector)

	local currentQValueVector, currentVValue = self:forwardPropagate(currentFeatureVector)

	local ClassesList = self.AdvantageModel:getClassesList()

	local actionIndex = table.find(ClassesList, action)

	local maxCurrentQValue = currentQValueVector[1][actionIndex]

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLossVector = AqwamMatrixLibrary:subtract(expectedQValue, previousQValue)

	local vLoss = currentVValue - previousVValue

	return qLossVector, vLoss

end

function ReinforcementLearningDeepDuelingQLearning:setAdvantageModel(AdvantageModel)
	
	self.AdvantageModel = AdvantageModel
	
end

function ReinforcementLearningDeepDuelingQLearning:setValueModel(ValueModel)

	self.ValueModel = ValueModel
	
end

function ReinforcementLearningDeepDuelingQLearning:getAdvantageModel()

	return self.AdvantageModel

end

function ReinforcementLearningDeepDuelingQLearning:getValueModel()

	return self.ValueModel

end

function ReinforcementLearningDeepDuelingQLearning:setUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningDeepDuelingQLearning:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningDeepDuelingQLearning:predict(featureVector, returnOriginalOutput)
	
	return self.AdvantageModel:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningDeepDuelingQLearning:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

end

function ReinforcementLearningDeepDuelingQLearning:episodeUpdate()

	local episodeUpdateFunction = self.episodeUpdateFunction

	if not episodeUpdateFunction then return end

	episodeUpdateFunction()

end

function ReinforcementLearningDeepDuelingQLearning:extendResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningDeepDuelingQLearning:reset()

	if (self.resetFunction) then self.resetFunction() end

end

function ReinforcementLearningDeepDuelingQLearning:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningDeepDuelingQLearning
