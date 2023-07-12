AdaptiveGradientDeltaOptimizer = {}

AdaptiveGradientDeltaOptimizer.__index = AdaptiveGradientDeltaOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultDecayRate = 0.9

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveGradientDeltaOptimizer.new(DecayRate, Epsilon)

	local NewAdaptiveGradientDeltaOptimizer = {}

	setmetatable(NewAdaptiveGradientDeltaOptimizer, AdaptiveGradientDeltaOptimizer)

	NewAdaptiveGradientDeltaOptimizer.RunningGradientSquaredMatrix = nil
	
	NewAdaptiveGradientDeltaOptimizer.RunningDeltaMatrix = nil
	
	NewAdaptiveGradientDeltaOptimizer.Epsilon = Epsilon or defaultEpsilon
	
	NewAdaptiveGradientDeltaOptimizer.DecayRate = DecayRate or defaultDecayRate

	return NewAdaptiveGradientDeltaOptimizer

end

function AdaptiveGradientDeltaOptimizer:calculate(learningRate, costFunctionDerivatives)

	self.RunningGradientSquaredMatrix = self.RunningGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	self.RunningDeltaMatrix = self.RunningDeltaMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

	local GradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)
	
	local DecayRunningGradientSquaredMatrix = AqwamMatrixLibrary:multiply(self.RunningGradientSquaredMatrix, self.DecayRate)
	
	local CurrentRunningGradientSquaredMatrix = AqwamMatrixLibrary:add(DecayRunningGradientSquaredMatrix, GradientSquaredMatrix)

	local DeltaMatrix = AqwamMatrixLibrary:subtract(CurrentRunningGradientSquaredMatrix, self.RunningGradientSquaredMatrix)
	
	local DecayRunningDeltaMatrix = AqwamMatrixLibrary:multiply(self.RunningDeltaMatrix, self.DecayRate)
	
	local CurrentRunningDeltaMatrix = AqwamMatrixLibrary:add(DecayRunningDeltaMatrix, DeltaMatrix)

	local SquareRootRunningDeltaMatrix = AqwamMatrixLibrary:power(CurrentRunningDeltaMatrix, 0.5)
	
	local EpsilonMatrix = AqwamMatrixLibrary:createMatrix(#SquareRootRunningDeltaMatrix, #SquareRootRunningDeltaMatrix[1], self.Epsilon)

	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, SquareRootRunningDeltaMatrix + EpsilonMatrix)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(costFunctionDerivativesPart1, -1)

	self.RunningGradientSquaredMatrix = CurrentRunningGradientSquaredMatrix
	
	self.RunningDeltaMatrix = CurrentRunningDeltaMatrix

	return costFunctionDerivatives

end

function AdaptiveGradientDeltaOptimizer:reset()

	self.PreviousSumOfGradientSquaredMatrix = nil

end

return AdaptiveGradientDeltaOptimizer
