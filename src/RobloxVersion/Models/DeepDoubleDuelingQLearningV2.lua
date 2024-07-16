local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DeepDoubleDuelingQLearning = {}

DeepDoubleDuelingQLearning.__index = DeepDoubleDuelingQLearning

local defaultAveragingRate = 0.01

local defaultDiscountFactor = 0.95

local function rateAverageModelParameters(averagingRate, PrimaryModelParameters, TargetModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local PrimaryModelParametersPart = AqwamMatrixLibrary:multiply(averagingRate, PrimaryModelParameters[layer])

		local TargetModelParametersPart = AqwamMatrixLibrary:multiply(averagingRateComplement, TargetModelParameters[layer])

		TargetModelParameters[layer] = AqwamMatrixLibrary:add(PrimaryModelParametersPart, TargetModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDoubleDuelingQLearning.new(averagingRate, discountFactor)

	local NewDeepDuelingQLearning = {}

	setmetatable(NewDeepDuelingQLearning, DeepDoubleDuelingQLearning)
	
	NewDeepDuelingQLearning.averagingRate = averagingRate or defaultDiscountFactor

	NewDeepDuelingQLearning.discountFactor = discountFactor or defaultDiscountFactor

	return NewDeepDuelingQLearning

end

function DeepDoubleDuelingQLearning:setParameters(averagingRate, discountFactor)
	
	self.averagingRate = averagingRate or self.averagingRate

	self.discountFactor =  discountFactor or self.discountFactor

end

function DeepDoubleDuelingQLearning:setAdvantageModel(Model)

	self.AdvantageModel = Model

end

function DeepDoubleDuelingQLearning:setValueModel(Model)

	self.ValueModel = Model

end

function DeepDoubleDuelingQLearning:forwardPropagate(featureVector)

	local vValue = self.ValueModel:predict(featureVector, true)[1][1]

	local advantageMatrix = self.AdvantageModel:predict(featureVector, true)

	local meanAdvantageVector = AqwamMatrixLibrary:horizontalMean(advantageMatrix)

	local qValuePart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValue = AqwamMatrixLibrary:add(vValue, qValuePart1)

	return qValue, vValue

end

function DeepDoubleDuelingQLearning:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local AdvantageModel = self.AdvantageModel
	
	local ValueModel = self.ValueModel
	
	local averagingRate = DeepDoubleDuelingQLearning.averagingRate
	
	if (AdvantageModel:getModelParameters() == nil) then AdvantageModel:generateLayers() end
	
	if (ValueModel:getModelParameters() == nil) then ValueModel:generateLayers() end

	local AdvantageModelPrimaryModelParameters = AdvantageModel:getModelParameters(true)
	
	local ValueModelPrimaryModelParameters = ValueModel:getModelParameters(true)

	local previousQValue, previousVValue = self:forwardPropagate(previousFeatureVector)

	local currentQValue, currentVValue = self:forwardPropagate(currentFeatureVector)

	local maxCurrentQValue = math.max(table.unpack(currentQValue[1]))

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLoss = AqwamMatrixLibrary:subtract(expectedQValue, previousQValue)

	local vLoss = AqwamMatrixLibrary:subtract(currentVValue, previousVValue)
	
	AdvantageModel:forwardPropagate(previousFeatureVector, true)

	AdvantageModel:backPropagate(qLoss, true)

	ValueModel:forwardPropagate(previousFeatureVector, true)

	ValueModel:backPropagate(vLoss, true)
	
	local AdvantageModelTargetModelParameters = AdvantageModel:getModelParameters(true)
	
	local ValueModelTargetModelParameters = AdvantageModel:getModelParameters(true)

	AdvantageModelTargetModelParameters = rateAverageModelParameters(averagingRate, AdvantageModelPrimaryModelParameters, AdvantageModelTargetModelParameters)
	
	ValueModelTargetModelParameters = rateAverageModelParameters(averagingRate, ValueModelPrimaryModelParameters, ValueModelTargetModelParameters)

	AdvantageModel:setModelParameters(AdvantageModelTargetModelParameters, true)
	
	ValueModel:setModelParameters(ValueModelTargetModelParameters, true)

	return vLoss

end

function DeepDoubleDuelingQLearning:predict(featureVector, returnOriginalOutput)

	return self.AdvantageModel:predict(featureVector, returnOriginalOutput)

end

function DeepDoubleDuelingQLearning:episodeUpdate()
	
	
end

function DeepDoubleDuelingQLearning:reset()
	
	
end

function DeepDoubleDuelingQLearning:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DeepDoubleDuelingQLearning
