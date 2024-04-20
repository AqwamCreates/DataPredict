local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DuelingQLearningModel = {}

DuelingQLearningModel.__index = DuelingQLearningModel

local defaultDiscountFactor = 0.95

function DuelingQLearningModel.new(discountFactor)

	local NewDuelingQLearningModel = {}

	setmetatable(NewDuelingQLearningModel, DuelingQLearningModel)

	NewDuelingQLearningModel.discountFactor = discountFactor or defaultDiscountFactor

	NewDuelingQLearningModel.ClassesList = nil

	return NewDuelingQLearningModel

end

function DuelingQLearningModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

function DuelingQLearningModel:setAdvantageModel(Model)

	self.AdvantageModel = Model

end

function DuelingQLearningModel:setValueModel(Model)

	self.ValueModel = Model

end

function DuelingQLearningModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function DuelingQLearningModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function DuelingQLearningModel:forwardPropagate(featureVector)

	local value = self.ValueModel:predict(featureVector, true)[1][1]

	local advantageMatrix = self.AdvantageModel:predict(featureVector, true)

	local meanAdvantageVector = AqwamMatrixLibrary:horizontalMean(advantageMatrix)

	local qValuePart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValue = AqwamMatrixLibrary:add(value, qValuePart1)

	return qValue, value

end

function DuelingQLearningModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local previousQValue, previousValue = self:forwardPropagate(previousFeatureVector)

	local currentQValue, currentValue = self:forwardPropagate(currentFeatureVector)

	local _, maxCurrentQValue = self:fetchHighestValueInVector(currentQValue)

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLoss = AqwamMatrixLibrary:subtract(expectedQValue, previousQValue)

	local vLoss = AqwamMatrixLibrary:subtract(currentValue, previousValue)
	
	self.AdvantageModel:forwardPropagate(previousFeatureVector, true)

	self.AdvantageModel:backPropagate(qLoss, true)

	self.ValueModel:forwardPropagate(previousFeatureVector, true)

	self.ValueModel:backPropagate(vLoss, true)

	return vLoss

end

function DuelingQLearningModel:predict(featureVector, returnOriginalOutput)

	return self.AdvantageModel:predict(featureVector, returnOriginalOutput)

end

function DuelingQLearningModel:episodeUpdate()
	
	
end

function DuelingQLearningModel:reset()
	
	
end

function DuelingQLearningModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DuelingQLearningModel
