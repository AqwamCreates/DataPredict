RootMeanSquarePropagationOptimizer = {}

RootMeanSquarePropagationOptimizer.__index = RootMeanSquarePropagationOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultBetaValue = 0.01

local defaultEpsilonValue = 0.01

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

function RootMeanSquarePropagationOptimizer:calculate(ModelParametersDerivatives)
	
	if (self.PreviousVelocityMatrix == nil) then 
		
		self.PreviousVelocityMatrix = AqwamMatrixLibrary:createMatrix(#ModelParametersDerivatives, #ModelParametersDerivatives[1])
		
	end
	
	local SquaredModelParameters = AqwamMatrixLibrary:power(ModelParametersDerivatives, 2)
	
	local VMatrixPart1 = AqwamMatrixLibrary:multiply(self.Beta, self.PreviousVelocityMatrix)
	
	local VMatrixPart2 = AqwamMatrixLibrary:multiply((1-self.Beta), SquaredModelParameters)
	
	local CurrentVelocityMatrix = AqwamMatrixLibrary:add(VMatrixPart1, VMatrixPart2)
	
	self.PreviousVelocityMatrix = CurrentVelocityMatrix
	
	local SquaredRootVelocityMatrix = AqwamMatrixLibrary:power(CurrentVelocityMatrix, 0.5)
	
	local DivisorMatrix = AqwamMatrixLibrary:add(SquaredRootVelocityMatrix, self.Epsilon)
	
	local RMSPropMatrix = AqwamMatrixLibrary:divide(ModelParametersDerivatives, DivisorMatrix)
	
	return RMSPropMatrix
	
end

function RootMeanSquarePropagationOptimizer:reset()
	
	self.PreviousVelocityMatrix = nil
	
end

return RootMeanSquarePropagationOptimizer

