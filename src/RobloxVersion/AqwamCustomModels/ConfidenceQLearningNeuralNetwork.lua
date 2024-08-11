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

ConfidenceQLearningNeuralNetwork = {}

ConfidenceQLearningNeuralNetwork.__index = ConfidenceQLearningNeuralNetwork

local defaultDiscountFactor = 0.95

-- Do not multiply confidenceValue with target! Otherwise, it will cause poor performance!

function ConfidenceQLearningNeuralNetwork.new(discountFactor)

	local NewConfidenceQLearningNeuralNetwork = {}
	
	setmetatable(NewConfidenceQLearningNeuralNetwork, ConfidenceQLearningNeuralNetwork)
	
	NewConfidenceQLearningNeuralNetwork.discountFactor =  discountFactor or defaultDiscountFactor

	return NewConfidenceQLearningNeuralNetwork

end

function ConfidenceQLearningNeuralNetwork:setParameters(discountFactor, confidenceLearningRate)
	
	self.discountFactor =  discountFactor or self.discountFactor

end

function ConfidenceQLearningNeuralNetwork:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function ConfidenceQLearningNeuralNetwork:setActorModel(ActorModel)
	
	self.ActorModel = ActorModel
	
end

function ConfidenceQLearningNeuralNetwork:getActorModel()
	
	return self.ActorModel

end

function ConfidenceQLearningNeuralNetwork:setConfidenceModel(ConfidenceModel)

	self.ConfidenceModel = ConfidenceModel

end

function ConfidenceQLearningNeuralNetwork:getConfidenceModel()

	return self.ConfidenceModel

end

function ConfidenceQLearningNeuralNetwork:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local ActorModel = self.ActorModel
	
	local ConfidenceModel = self.ConfidenceModel

	local previousQVector = ActorModel:predict(previousFeatureVector, true)
	
	local currentQVector = ActorModel:predict(currentFeatureVector, true)

	local currentMaxQValue = math.max(table.unpack(currentQVector[1]))

	--local qValueRatio = maxQValue / previousMaxQValue
	
	local ClassesList = ActorModel:getClassesList()

	local numberOfClasses = #ClassesList
	
	local previousConfidence = ConfidenceModel:predict(previousFeatureVector, true)[1][1]

	local currentConfidence = ConfidenceModel:predict(currentFeatureVector, true)[1][1]

	--local relativeChange = (currentConfidence - previousConfidence) / (currentConfidence + previousConfidence)

	local relativeChange = (currentConfidence - previousConfidence) / previousConfidence

	--local relativeChange = (currentConfidence - previousConfidence) / previousConfidence -- Doesn't work well.

	--relativeChange = math.clamp(relativeChange, -10, 10)

	local qValue = currentMaxQValue * relativeChange

	--print(previousConfidence)

	--print(relativeChange)

	--relativeConfidence = math.clamp(relativeConfidence, -10, 10)

	local target = (rewardValue) + (self.discountFactor * relativeChange)

	--local targetVector = AqwamMatrixLibrary:multiply(NewConfidenceQLearningNeuralNetwork.discountFactor, relativeChange, currentQVector)

	--targetVector = AqwamMatrixLibrary:add(targetVector, rewardValue)

	local actionIndex = table.find(ClassesList, action)

	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)
	
	local temporalDifferenceError = target - previousQVector[1][actionIndex]

	--local lossVector = AqwamMatrixLibrary:subtract(targetVector, previousQVector)

	lossVector[1][actionIndex] = temporalDifferenceError

	ActorModel:forwardPropagate(previousFeatureVector, true)

	ActorModel:backwardPropagate(lossVector, true)
	
	ConfidenceModel:forwardPropagate(previousFeatureVector, true)
	
	ConfidenceModel:backwardPropagate(rewardValue)
	
	return temporalDifferenceError
	
end

function ConfidenceQLearningNeuralNetwork:episodeUpdate()
		
end

function ConfidenceQLearningNeuralNetwork:reset()
	
end

return ConfidenceQLearningNeuralNetwork