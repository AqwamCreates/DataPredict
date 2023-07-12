AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

function AdaptiveMomentEstimationOptimizer.new(Beta1, Beta2)

	local NewAdaptiveMomentEstimationOptimizer = {}

	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)

	NewAdaptiveMomentEstimationOptimizer.PreviousMomentum = nil
	
	NewAdaptiveMomentEstimationOptimizer.PreviousVelocity = nil
	
	NewAdaptiveMomentEstimationOptimizer.Beta1 = Beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.Beta2 = Beta2 or defaultBeta2

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(Beta1)
	
	self.Beta1 = Beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(Beta2)
		
	self.Beta2 = Beta2
	
end


function AdaptiveMomentEstimationOptimizer:calculate(costFunctionDerivatives)

	self.PreviousMomentum = self.PreviousMomentum or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	self.PreviousVelocity = self.PreviousVelocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

	local momentumPart1 = AqwamMatrixLibrary:multiply(self.Beta1, self.PreviousMomentum)
	
	local momentumPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta1), costFunctionDerivatives)
	
	local momentum = AqwamMatrixLibrary:add(momentumPart1, momentumPart2)

	local squaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

	local velocityPart1 = AqwamMatrixLibrary:multiply(self.Beta2, self.PreviousVelocity)
	
	local velocityPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta2), squaredModelParameters)
	
	local velocity = AqwamMatrixLibrary:add(velocityPart1, velocityPart2)

	local meanMomentum = AqwamMatrixLibrary:divide(momentum, (1 - self.Beta1))
	
	local meanVelocity = AqwamMatrixLibrary:divide(velocity, (1 - self.Beta2))
	
	local squareRootedDivisor = AqwamMatrixLibrary:power(meanVelocity, 0.5)

	local costFunctionDerivatives = AqwamMatrixLibrary:divide(meanMomentum, squareRootedDivisor)

	self.PreviousMomentum = momentum
	
	self.PreviousVelocity = velocity

	return costFunctionDerivatives
end

function AdaptiveMomentEstimationOptimizer:reset()

	self.PreviousMomentum = nil

	self.PreviousVelocity = nil

end

return AdaptiveMomentEstimationOptimizer
