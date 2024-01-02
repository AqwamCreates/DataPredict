local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

AdaptiveMomentEstimationMaximumOptimizer = {}

AdaptiveMomentEstimationMaximumOptimizer.__index = AdaptiveMomentEstimationMaximumOptimizer

setmetatable(AdaptiveMomentEstimationMaximumOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationMaximumOptimizer.new(beta1, beta2, epsilon)

	local NewAdaptiveMomentEstimationMaximumOptimizer = BaseOptimizer.new("AdaptiveMomentEstimationMaximum")

	setmetatable(NewAdaptiveMomentEstimationMaximumOptimizer, AdaptiveMomentEstimationMaximumOptimizer)

	
	NewAdaptiveMomentEstimationMaximumOptimizer.beta1 = beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationMaximumOptimizer.beta2 = beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationMaximumOptimizer.epsilon = epsilon or defaultEpsilon
	
	NewAdaptiveMomentEstimationMaximumOptimizer.timeStep = 0
	
	NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight = nil
	
	NewAdaptiveMomentEstimationMaximumOptimizer.moment = nil
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)

		NewAdaptiveMomentEstimationMaximumOptimizer.moment = NewAdaptiveMomentEstimationMaximumOptimizer.moment or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight = NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		NewAdaptiveMomentEstimationMaximumOptimizer.timeStep += 1

		local MomentPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveMomentEstimationMaximumOptimizer.beta1, NewAdaptiveMomentEstimationMaximumOptimizer.moment)

		local MomentPart2 = AqwamMatrixLibrary:multiply((1 - NewAdaptiveMomentEstimationMaximumOptimizer.beta1), costFunctionDerivatives)

		NewAdaptiveMomentEstimationMaximumOptimizer.moment = AqwamMatrixLibrary:add(MomentPart1, MomentPart2)

		local ExponentWeightPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveMomentEstimationMaximumOptimizer.beta2, NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight)

		local ExponentWeightPart2 = AqwamMatrixLibrary:applyFunction(math.abs, costFunctionDerivatives)

		NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight = AqwamMatrixLibrary:applyFunction(math.max, ExponentWeightPart1, ExponentWeightPart2)

		local DivisorPart1 = 1 - math.pow(NewAdaptiveMomentEstimationMaximumOptimizer.beta1, 2)

		local DivisorPart2 = AqwamMatrixLibrary:add(NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight, NewAdaptiveMomentEstimationMaximumOptimizer.epsilon)

		local Divisor = AqwamMatrixLibrary:multiply(DivisorPart1, DivisorPart1)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(NewAdaptiveMomentEstimationMaximumOptimizer.moment, Divisor)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		return costFunctionDerivatives

	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setResetFunction(function()
	
		NewAdaptiveMomentEstimationMaximumOptimizer.timeStep = 0

		NewAdaptiveMomentEstimationMaximumOptimizer.exponentWeight = nil

		NewAdaptiveMomentEstimationMaximumOptimizer.moment = nil
		
	end)

	return NewAdaptiveMomentEstimationMaximumOptimizer

end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationMaximumOptimizer
