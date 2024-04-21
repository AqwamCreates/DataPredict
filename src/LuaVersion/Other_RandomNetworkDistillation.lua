local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local NeuralNetworkModel = require("Model_NeuralNetwork")

local RandomNetworkDistillation = {}

RandomNetworkDistillation.__index = RandomNetworkDistillation

setmetatable(RandomNetworkDistillation, NeuralNetworkModel)

local defaultMaxNumberOfIterations = 1

function RandomNetworkDistillation.new(maxNumberOfIterations)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	local NewRandomNetworkDistillation = NeuralNetworkModel.new(maxNumberOfIterations)
	
	setmetatable(NewRandomNetworkDistillation, RandomNetworkDistillation)
	
	NewRandomNetworkDistillation.TargetModelParameters = nil
	
	NewRandomNetworkDistillation.PredictorModelParameters = nil
	
	return NewRandomNetworkDistillation
	
end

function RandomNetworkDistillation:setParameters(maxNumberOfIterations)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
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

function RandomNetworkDistillation:generate(featureVector)
	
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
	
	local value = math.sqrt(sumError)
	
	local numberOfFeatures = #featureVector[1]
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

	self:forwardPropagate(featureVector, true)
	self:backPropagate(errorVector, true)
	
	return value
	
end

function RandomNetworkDistillation:getTargetModelParameters()
	
	return self.TargetModelParameters 
	
end

function RandomNetworkDistillation:getPredictorModelParameters()
	
	return self.PredictorModelParameters
	
end

function RandomNetworkDistillation:setTargetModelParameters(TargetModelParameters)
	
	self.TargetModelParameters = TargetModelParameters
	
end

function RandomNetworkDistillation:setPredictorModelParameters(PredictorModelParameters)

	self.PredictorModelParameters = PredictorModelParameters

end

return RandomNetworkDistillation
