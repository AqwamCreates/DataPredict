NesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

NesterovAcceleratedAdaptiveMomentEstimationOptimizer.__index = NesterovAcceleratedAdaptiveMomentEstimationOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer.new(Beta1, Beta2)

	local NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

	setmetatable(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer, NesterovAcceleratedAdaptiveMomentEstimationOptimizer)

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.PreviousM = nil
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.PreviousN = nil
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.Beta1 = Beta1 or defaultBeta1
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.Beta2 = Beta2 or defaultBeta2

	return NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta1(Beta1)
	
	self.Beta1 = Beta1
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta2(Beta2)
		
	self.Beta2 = Beta2
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	self.PreviousM = self.PreviousM or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

	self.PreviousN = self.PreviousN or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	local meanCostFunctionDerivatives = AqwamMatrixLibrary:divide(costFunctionDerivatives, (1 - self.Beta1))
	
	local MPart1 = AqwamMatrixLibrary:multiply(self.Beta1, self.PreviousM)
	
	local MPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta1), costFunctionDerivatives)
	
	local M = AqwamMatrixLibrary:add(MPart1, MPart2)
	
	local meanM = AqwamMatrixLibrary:divide(M, (1 - self.Beta1))

	local squaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

	local NPart1 = AqwamMatrixLibrary:multiply(self.Beta2, self.PreviousN)
	
	local NPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta2), squaredModelParameters)
	
	local N = AqwamMatrixLibrary:add(NPart1, NPart2)
	
	local meanN = AqwamMatrixLibrary:divide(N, (1 - self.Beta2))
	
	local finalMPart1 = AqwamMatrixLibrary:multiply((1 - self.Beta1), meanCostFunctionDerivatives)
	
	local finalMPart2 = AqwamMatrixLibrary:multiply(self.Beta1, meanM)
	
	local finalM = AqwamMatrixLibrary:add(finalMPart1, finalMPart2)
	
	local squareRootedDivisor = AqwamMatrixLibrary:power(meanN, 0.5)

	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(finalM, squareRootedDivisor)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

	self.PreviousM = M
	
	self.PreviousN = N

	return costFunctionDerivatives
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:reset()

	self.PreviousM = nil

	self.PreviousN = nil

end

return NesterovAcceleratedAdaptiveMomentEstimationOptimizer
