AdaptiveMomentEstimationMaximumOptimizer = {}

AdaptiveMomentEstimationMaximumOptimizer.__index = AdaptiveMomentEstimationMaximumOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationMaximumOptimizer.new(Beta1, Beta2, Epsilon)

	local NewAdaptiveMomentEstimationMaximumOptimizer = {}

	setmetatable(NewAdaptiveMomentEstimationMaximumOptimizer, AdaptiveMomentEstimationMaximumOptimizer)

	NewAdaptiveMomentEstimationMaximumOptimizer.PreviousMomentum = nil
	
	NewAdaptiveMomentEstimationMaximumOptimizer.PreviousVelocity = nil
	
	NewAdaptiveMomentEstimationMaximumOptimizer.Beta1 = Beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationMaximumOptimizer.Beta2 = Beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationMaximumOptimizer.Epsilon = Epsilon or defaultEpsilon
	
	NewAdaptiveMomentEstimationMaximumOptimizer.TimeStep = 0
	
	NewAdaptiveMomentEstimationMaximumOptimizer.ExponentWeight = nil
	
	NewAdaptiveMomentEstimationMaximumOptimizer.Moment = nil

	return NewAdaptiveMomentEstimationMaximumOptimizer

end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta1(Beta1)
	
	self.Beta1 = Beta1
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta2(Beta2)
		
	self.Beta2 = Beta2
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setEpsilon(Epsilon)

	self.Epsilon = Epsilon

end

function AdaptiveMomentEstimationMaximumOptimizer:calculate(learningRate, costFunctionDerivatives)

	self.Moment = self.Moment or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	self.ExponentWeight = self.ExponentWeight or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	self.TimeStep += 1
	
	local MomentPart1 = AqwamMatrixLibrary:multiply(self.Beta1, self.Moment)
	
	local MomentPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta1), costFunctionDerivatives)
	
	self.Moment = AqwamMatrixLibrary:add(MomentPart1, MomentPart2)
	
	local ExponentWeightPart1 = AqwamMatrixLibrary:multiply(self.Beta2, self.ExponentWeight)
	
	local ExponentWeightPart2 = AqwamMatrixLibrary:applyFunction(math.abs, costFunctionDerivatives)
	
	self.ExponentWeight = AqwamMatrixLibrary:applyFunction(math.max, ExponentWeightPart1, ExponentWeightPart2)
	
	local DivisorPart1 = 1 - math.pow(self.Beta1, 2)
	
	local DivisorPart2 = AqwamMatrixLibrary:add(self.ExponentWeight, self.Epsilon)
	
	local Divisor = AqwamMatrixLibrary:multiply(DivisorPart1, DivisorPart1)
	
	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(self.Moment, Divisor)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

	return costFunctionDerivatives
	
end

function AdaptiveMomentEstimationMaximumOptimizer:reset()

	self.TimeStep = 0

	self.ExponentWeight = nil

	self.Moment = nil
	
end

return AdaptiveMomentEstimationMaximumOptimizer
