local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local RandomNetworkDistillation = {}

RandomNetworkDistillation.__index = RandomNetworkDistillation

local defaultMaxNumberOfIterations = 1

function RandomNetworkDistillation.new()
	
	local NewRandomNetworkDistillation = {}
	
	setmetatable(NewRandomNetworkDistillation, RandomNetworkDistillation)
	
	NewRandomNetworkDistillation.Model = nil
	
	NewRandomNetworkDistillation.TargetModelParameters = nil
	
	NewRandomNetworkDistillation.PredictorModelParameters = nil
	
	return NewRandomNetworkDistillation
	
end

function RandomNetworkDistillation:setModel(Model)
	
	self.Model = Model
	
end

function RandomNetworkDistillation:getModel(Model)
	
	return self.Model
	
end

function RandomNetworkDistillation:generateModelParameters()
	
	local Model = self.Model
	
	if (not self.TargetModelParameters) then
		
		Model:generateLayers()
		
		self.TargetModelParameters = Model:getModelParameters(true)
		
	end
	
	if (not self.PredictorModelParameters) then

		Model:generateLayers()

		self.PredictorModelParameters = Model:getModelParameters(true)

	end
	
end

function RandomNetworkDistillation:generate(featureVector)
	
	if (not self.TargetModelParameters) or (not self.PredictorModelParameters) then
		
		self:generateModelParameters()
		
	end
	
	local Model = self.Model
	
	self:setModelParameters(self.TargetModelParameters, true)
	
	local targetVector = Model:predict(featureVector, true)
	
	self:setModelParameters(self.PredictorModelParameters, true)

	local predictorVector = Model:predict(featureVector, true)
	
	local errorVector = AqwamMatrixLibrary:subtract(predictorVector, targetVector)
	
	local squaredErrorVector = AqwamMatrixLibrary:power(errorVector, 2)
	
	local sumError = AqwamMatrixLibrary:sum(squaredErrorVector)
	
	local value = math.sqrt(sumError)
	
	local numberOfFeatures = #featureVector[1]
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

	Model:forwardPropagate(featureVector, true)
	Model:backPropagate(errorVector, true)
	
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
