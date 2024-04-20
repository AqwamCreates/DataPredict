local BaseOptimizer = require("Optimizer_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

setmetatable(AdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationOptimizer.new(beta1, beta2, epsilon)

	local NewAdaptiveMomentEstimationOptimizer = BaseOptimizer.new("AdaptiveMomentEstimation")

	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)

	NewAdaptiveMomentEstimationOptimizer.previousMomentum = nil
	
	NewAdaptiveMomentEstimationOptimizer.previousVelocity = nil
	
	NewAdaptiveMomentEstimationOptimizer.beta1 = beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.beta2 = beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationOptimizer.epsilon = epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveMomentEstimationOptimizer.previousMomentum = NewAdaptiveMomentEstimationOptimizer.previousMomentum or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		NewAdaptiveMomentEstimationOptimizer.previousVelocity = NewAdaptiveMomentEstimationOptimizer.previousVelocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local momentumPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveMomentEstimationOptimizer.beta1, NewAdaptiveMomentEstimationOptimizer.previousMomentum)

		local momentumPart2 = AqwamMatrixLibrary:multiply((1 - NewAdaptiveMomentEstimationOptimizer.beta1), costFunctionDerivatives)

		local momentum = AqwamMatrixLibrary:add(momentumPart1, momentumPart2)

		local squaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local velocityPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveMomentEstimationOptimizer.beta2, NewAdaptiveMomentEstimationOptimizer.PreviousVelocity)

		local velocityPart2 = AqwamMatrixLibrary:multiply((1 - NewAdaptiveMomentEstimationOptimizer.beta2), squaredModelParameters)

		local velocity = AqwamMatrixLibrary:add(velocityPart1, velocityPart2)

		local meanMomentum = AqwamMatrixLibrary:divide(momentum, (1 - NewAdaptiveMomentEstimationOptimizer.beta1))

		local meanVelocity = AqwamMatrixLibrary:divide(velocity, (1 - NewAdaptiveMomentEstimationOptimizer.beta2))

		local squareRootedDivisor = AqwamMatrixLibrary:power(meanVelocity, 0.5)

		local finalDivisor = AqwamMatrixLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationOptimizer.Epsilon)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(meanMomentum, finalDivisor)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveMomentEstimationOptimizer.previousMomentum = momentum

		NewAdaptiveMomentEstimationOptimizer.previousVelocity = velocity

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setResetFunction(function()
		
		NewAdaptiveMomentEstimationOptimizer.previousMomentum = nil

		NewAdaptiveMomentEstimationOptimizer.previousVelocity = nil
		
	end)

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationOptimizer
