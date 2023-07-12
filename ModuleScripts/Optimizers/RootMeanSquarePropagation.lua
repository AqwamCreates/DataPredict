RootMeanSquarePropagationOptimizer = {}

RootMeanSquarePropagationOptimizer.__index = RootMeanSquarePropagationOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultBetaValue = 0.1

local defaultEpsilonValue = 1 * math.pow(10, -7)

function RootMeanSquarePropagationOptimizer.new(Beta, Epsilon)
	
	local NewRootMeanSquarePropagationOptimizer = {}
	
	setmetatable(NewRootMeanSquarePropagationOptimizer, RootMeanSquarePropagationOptimizer)
	
	NewRootMeanSquarePropagationOptimizer.Beta = Beta or defaultBetaValue
	
	NewRootMeanSquarePropagationOptimizer.Epsilon = Epsilon or defaultEpsilonValue
	
	NewRootMeanSquarePropagationOptimizer.PreviousVelocityMatrix = nil
	
	return NewRootMeanSquarePropagationOptimizer
	
end

function RootMeanSquarePropagationOptimizer:setBeta(Beta)
	
	self.Beta = Beta
	
end

function RootMeanSquarePropagationOptimizer:setEpsilon(Epsilon)

	self.Epsilon = Epsilon

end

function RootMeanSquarePropagationOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	self.PreviousVelocityMatrix = self.PreviousVelocityMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	local SquaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)
	
	local VMatrixPart1 = AqwamMatrixLibrary:multiply(self.Beta, self.PreviousVelocityMatrix)
	
	local VMatrixPart2 = AqwamMatrixLibrary:multiply((1-self.Beta), SquaredModelParameters)
	
	local CurrentVelocityMatrix = AqwamMatrixLibrary:add(VMatrixPart1, VMatrixPart2)
	
	local NonZeroDivisorMatrix = AqwamMatrixLibrary:add(CurrentVelocityMatrix, self.Epsilon)
	
	local SquaredRootVelocityMatrix = AqwamMatrixLibrary:power(NonZeroDivisorMatrix, 0.5)
	
	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, SquaredRootVelocityMatrix)
	
	local costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)
	
	self.PreviousVelocityMatrix = CurrentVelocityMatrix
	
	return costFunctionDerivatives
	
end

function RootMeanSquarePropagationOptimizer:reset()
	
	self.PreviousVelocityMatrix = nil
	
end

return RootMeanSquarePropagationOptimizer
