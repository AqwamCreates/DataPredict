local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local NeuralNetworkModel = require(script.Parent.Parent.Models.NeuralNetwork)

local RandomNetworkDistillation = {}

RandomNetworkDistillation.__index = RandomNetworkDistillation

setmetatable(RandomNetworkDistillation, NeuralNetworkModel)

local defaultMaxNumberOfIterations = 1

function RandomNetworkDistillation.new(maxNumberOfIterations, learningRate)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	local NewRandomNetworkDistillation = NeuralNetworkModel.new(maxNumberOfIterations, learningRate)
	
	setmetatable(NewRandomNetworkDistillation, RandomNetworkDistillation)
	
	NewRandomNetworkDistillation.TargetModelParameters = nil
	
	NewRandomNetworkDistillation.PredictorModelParameters = nil
	
	return NewRandomNetworkDistillation
	
end

function RandomNetworkDistillation:setParameters(maxNumberOfIterations, learningRate)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.learningRate = learningRate or self.learningRate
	
end

function RandomNetworkDistillation:generateModelParameters()
	
	if (not self.TargetModelParameters) then
		
		self:generateLayers()
		
		self.TargetModelParameters = self:getModelParameters(true)
		
	end
	
	if (not self.PredictorModelParameters) then

		self:generateLayers()

		self.PredictorModelParameters = self:getModelParameters(true)

	end
	
end

function RandomNetworkDistillation:generateReward(featureVector)
	
	if (not self.TargetModelParameters) or (not self.PredictorModelParameters) then
		
		self:generateModelParameters()
		
	end
	
	self:setModelParameters(self.TargetModelParameters, true)
	
	local targetVector = self:predict(featureVector, true)
	
	self:setModelParameters(self.PredictorModelParameters, true)

	local predictorVector = self:predict(featureVector, true)
	
	local errorVector = AqwamMatrixLibrary:subtract(predictorVector, targetVector)
	
	local squaredErrorVector = AqwamMatrixLibrary:power(errorVector, 2)
	
	local sumError = AqwamMatrixLibrary:sum(squaredErrorVector)
	
	local reward = math.sqrt(sumError)
	
	local numberOfFeatures = #featureVector[1]
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

	self:forwardPropagate(featureVector, true)
	self:backPropagate(errorVector, true)
	
	return reward
	
end

function RandomNetworkDistillation:getTargetModelParameters()
	
	return self.TargetModelParameters 
	
end

function RandomNetworkDistillation:getPredictorModelParameters()
	
	return self.PredictorModelParameters
	
end

function RandomNetworkDistillation:setTargetModelParameters(TargetModelParameters)
	
	self.TargetModelParameters  = TargetModelParameters
	
end

function RandomNetworkDistillation:setPredictorModelParameters(PredictorModelParameters)

	self.PredictorModelParameters  = PredictorModelParameters

end

return RandomNetworkDistillation
