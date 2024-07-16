local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DeepDuelingQLearning = {}

DeepDuelingQLearning.__index = DeepDuelingQLearning

local defaultDiscountFactor = 0.95

function DeepDuelingQLearning.new(discountFactor)

	local NewDeepDuelingQLearning = {}

	setmetatable(NewDeepDuelingQLearning, DeepDuelingQLearning)

	NewDeepDuelingQLearning.discountFactor = discountFactor or defaultDiscountFactor

	return NewDeepDuelingQLearning

end

function DeepDuelingQLearning:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

function DeepDuelingQLearning:setAdvantageModel(Model)

	self.AdvantageModel = Model

end

function DeepDuelingQLearning:setValueModel(Model)

	self.ValueModel = Model

end

function DeepDuelingQLearning:forwardPropagate(featureVector)

	local vValue = self.ValueModel:predict(featureVector, true)[1][1]

	local advantageMatrix = self.AdvantageModel:predict(featureVector, true)

	local meanAdvantageVector = AqwamMatrixLibrary:horizontalMean(advantageMatrix)

	local qValuePart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValue = AqwamMatrixLibrary:add(vValue, qValuePart1)

	return qValue, vValue

end

function DeepDuelingQLearning:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local previousQValue, previousVValue = self:forwardPropagate(previousFeatureVector)

	local currentQValue, currentVValue = self:forwardPropagate(currentFeatureVector)

	local maxCurrentQValue = math.max(table.unpack(currentQValue[1]))

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLossVector = AqwamMatrixLibrary:subtract(expectedQValue, previousQValue)

	local vLoss = currentVValue - previousVValue
	
	self.AdvantageModel:forwardPropagate(previousFeatureVector, true)

	self.AdvantageModel:backPropagate(qLossVector, true)

	self.ValueModel:forwardPropagate(previousFeatureVector, true)

	self.ValueModel:backPropagate(vLoss, true)

	return vLoss

end

function DeepDuelingQLearning:predict(featureVector, returnOriginalOutput)

	return self.AdvantageModel:predict(featureVector, returnOriginalOutput)

end

function DeepDuelingQLearning:episodeUpdate()
	
	
end

function DeepDuelingQLearning:reset()
	
	
end

function DeepDuelingQLearning:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DeepDuelingQLearning
