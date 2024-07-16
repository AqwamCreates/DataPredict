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

	local qValueVectorPart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValueVector = AqwamMatrixLibrary:add(vValue, qValueVectorPart1)

	return qValueVector, vValue

end

function DeepDuelingQLearning:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local AdvantageModel = self.AdvantageModel
	
	local ValueModel = self.ValueModel

	local previousQValueVector, previousVValue = self:forwardPropagate(previousFeatureVector)

	local currentQValueVector, currentVValue = self:forwardPropagate(currentFeatureVector)
	
	local ClassesList = AdvantageModel:getClassesList()

	local numberOfClasses = #ClassesList
	
	local actionIndex = table.find(ClassesList, action)

	local maxCurrentQValue = currentQValueVector[1][actionIndex]

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLossVector = AqwamMatrixLibrary:subtract(expectedQValue, previousQValueVector)

	local vLoss = currentVValue - previousVValue
	
	AdvantageModel:forwardPropagate(previousFeatureVector, true)

	AdvantageModel:backPropagate(qLossVector, true)

	ValueModel:forwardPropagate(previousFeatureVector, true)

	ValueModel:backPropagate(vLoss, true)

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
