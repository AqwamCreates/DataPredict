local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

NesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

NesterovAcceleratedAdaptiveMomentEstimationOptimizer.__index = NesterovAcceleratedAdaptiveMomentEstimationOptimizer

setmetatable(NesterovAcceleratedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer.new(beta1, beta2, epsilon)

	local NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new("NesterovAcceleratedAdaptiveMomentEstimation")

	setmetatable(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer, NesterovAcceleratedAdaptiveMomentEstimationOptimizer)
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1 = beta1 or defaultBeta1

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2 = beta2 or defaultBeta2

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon = epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local previousM

		local previousN
		
		local optimizerInternalParameters = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameters
		
		if (optimizerInternalParameters) then
			
			previousM = optimizerInternalParameters[1]
			
			previousN = optimizerInternalParameters[2]
			
		end
		
		previousM = previousM or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		previousN = previousN or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		local beta1 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1
		
		local beta2 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2

		local meanCostFunctionDerivatives = AqwamMatrixLibrary:divide(costFunctionDerivatives, (1 - beta1))

		local mPart1 = AqwamMatrixLibrary:multiply(beta1, previousM)

		local mPart2 = AqwamMatrixLibrary:multiply((1 - beta1), costFunctionDerivatives)

		local m = AqwamMatrixLibrary:add(mPart1, mPart2)

		local meanM = AqwamMatrixLibrary:divide(m, (1 - beta1))

		local squaredCostFunctionDerivatives = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local nPart1 = AqwamMatrixLibrary:multiply(beta2, previousN)

		local nPart2 = AqwamMatrixLibrary:multiply((1 - beta2), squaredCostFunctionDerivatives)

		local n = AqwamMatrixLibrary:add(nPart1, nPart2)

		local meanN = AqwamMatrixLibrary:divide(n, (1 - beta2))

		local finalMPart1 = AqwamMatrixLibrary:multiply((1 - beta1), meanCostFunctionDerivatives)

		local finalMPart2 = AqwamMatrixLibrary:multiply(beta1, meanM)

		local finalM = AqwamMatrixLibrary:add(finalMPart1, finalMPart2)

		local squareRootedDivisor = AqwamMatrixLibrary:power(meanN, 0.5)

		local finalDivisor = AqwamMatrixLibrary:add(squareRootedDivisor, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(finalM, finalDivisor)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameters = {m, n}

		return costFunctionDerivatives
		
	end)
	
	return NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return NesterovAcceleratedAdaptiveMomentEstimationOptimizer
