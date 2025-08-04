--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

DeepConfidenceQLearningModel = {}

DeepConfidenceQLearningModel.__index = DeepConfidenceQLearningModel

local defaultDiscountFactor = 0.95

-- Do not multiply confidenceValue with target! Otherwise, it will cause poor performance!

function DeepConfidenceQLearningModel.new(discountFactor)

	local NewDeepConfidenceQLearningModel = {}
	
	setmetatable(NewDeepConfidenceQLearningModel, DeepConfidenceQLearningModel)
	
	NewDeepConfidenceQLearningModel.discountFactor =  discountFactor or defaultDiscountFactor

	return NewDeepConfidenceQLearningModel

end

function DeepConfidenceQLearningModel:setParameters(discountFactor, confidenceLearningRate)
	
	self.discountFactor =  discountFactor or self.discountFactor

end

function DeepConfidenceQLearningModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function DeepConfidenceQLearningModel:setActorModel(ActorModel)
	
	self.ActorModel = ActorModel
	
end

function DeepConfidenceQLearningModel:getActorModel()
	
	return self.ActorModel

end

function DeepConfidenceQLearningModel:setConfidenceModel(ConfidenceModel)

	self.ConfidenceModel = ConfidenceModel

end

function DeepConfidenceQLearningModel:getConfidenceModel()

	return self.ConfidenceModel

end

function DeepConfidenceQLearningModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local ActorModel = self.ActorModel
	
	local ConfidenceModel = self.ConfidenceModel

	local previousQVector = ActorModel:predict(previousFeatureVector, true)
	
	local currentQVector = ActorModel:predict(currentFeatureVector, true)

	local currentMaxQValue = math.max(table.unpack(currentQVector[1]))

	--local qValueRatio = maxQValue / previousMaxQValue
	
	local ClassesList = ActorModel:getClassesList()

	local numberOfClasses = #ClassesList
	
	local previousConfidence = ConfidenceModel:predict(previousQVector, true)[1][1]

	local currentConfidence = ConfidenceModel:predict(currentQVector, true)[1][1]

	--local relativeChange = (currentConfidence - previousConfidence) / (currentConfidence + previousConfidence)

	local relativeConfidence = (currentConfidence - previousConfidence) / previousConfidence

	--local relativeChange = (currentConfidence - previousConfidence) / previousConfidence -- Doesn't work well.

	--relativeChange = math.clamp(relativeChange, -10, 10)

	--print(previousConfidence)

	--print(relativeChange)

	--relativeConfidence = math.clamp(relativeConfidence, -10, 10)

	local target = (rewardValue * relativeConfidence) + (self.discountFactor * currentMaxQValue)

	--local targetVector = AqwamMatrixLibrary:multiply(NewConfidenceQLearningNeuralNetwork.discountFactor, relativeChange, currentQVector)

	--targetVector = AqwamMatrixLibrary:add(targetVector, rewardValue)

	local actionIndex = table.find(ClassesList, action)

	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)
	
	local temporalDifferenceError = target - previousQVector[1][actionIndex]

	--local lossVector = AqwamMatrixLibrary:subtract(targetVector, previousQVector)

	lossVector[1][actionIndex] = temporalDifferenceError

	ActorModel:forwardPropagate(previousFeatureVector, true)

	ActorModel:backwardPropagate(lossVector, true)
	
	ConfidenceModel:forwardPropagate(previousQVector, true)
	
	ConfidenceModel:backwardPropagate(rewardValue)
	
	return temporalDifferenceError
	
end

function DeepConfidenceQLearningModel:diagonalGaussianUpdate()
	
	error("The diagonal Gaussian update is not implemented!")
	
end

function DeepConfidenceQLearningModel:episodeUpdate()
		
end

function DeepConfidenceQLearningModel:reset()
	
end

return DeepConfidenceQLearningModel
