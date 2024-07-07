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
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)

		local moment

		local exponentWeight
		
		local optimizerInternalParameters = NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameters
		
		if (optimizerInternalParameters) then
			
			moment = optimizerInternalParameters[1]
			
			exponentWeight = optimizerInternalParameters[2]
			
		end
		
		moment = moment or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		exponentWeight = exponentWeight or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		local beta1 = NewAdaptiveMomentEstimationMaximumOptimizer.beta1
		
		local beta2 = NewAdaptiveMomentEstimationMaximumOptimizer.beta2

		local momentPart1 = AqwamMatrixLibrary:multiply(beta1, moment)

		local momentPart2 = AqwamMatrixLibrary:multiply((1 - beta1), costFunctionDerivatives)

		moment = AqwamMatrixLibrary:add(momentPart1, momentPart2)

		local exponentWeightPart1 = AqwamMatrixLibrary:multiply(beta2, exponentWeight)

		local exponentWeightPart2 = AqwamMatrixLibrary:applyFunction(math.abs, costFunctionDerivatives)

		exponentWeight = AqwamMatrixLibrary:applyFunction(math.max, exponentWeightPart1, exponentWeightPart2)

		local divisorPart1 = 1 - math.pow(beta1, 2)

		local divisorPart2 = AqwamMatrixLibrary:add(exponentWeight, NewAdaptiveMomentEstimationMaximumOptimizer.epsilon)

		local divisor = AqwamMatrixLibrary:multiply(divisorPart1, divisorPart2)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(moment, divisor)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)
		
		NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameters = {moment, exponentWeight}

		return costFunctionDerivatives

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
