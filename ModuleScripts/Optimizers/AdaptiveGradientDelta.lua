AdaptiveGradientDeltaOptimizer = {}

AdaptiveGradientDeltaOptimizer.__index = AdaptiveGradientDeltaOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultDecayRate = 0.9

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveGradientDeltaOptimizer.new(DecayRate, Epsilon)

	local NewAdaptiveGradientDeltaOptimizer = {}

	setmetatable(NewAdaptiveGradientDeltaOptimizer, AdaptiveGradientDeltaOptimizer)

	NewAdaptiveGradientDeltaOptimizer.PreviousRunningGradientSquaredMatrix = nil
	
	NewAdaptiveGradientDeltaOptimizer.Epsilon = Epsilon or defaultEpsilon
	
	NewAdaptiveGradientDeltaOptimizer.DecayRate = DecayRate or defaultDecayRate

	return NewAdaptiveGradientDeltaOptimizer

end

function AdaptiveGradientDeltaOptimizer:calculate(learningRate, costFunctionDerivatives)

	self.PreviousRunningGradientSquaredMatrix = self.PreviousRunningGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

	local GradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)
	
	local RunningDeltaMatrixPart1 = AqwamMatrixLibrary:multiply(self.DecayRate, self.PreviousRunningGradientSquaredMatrix)
	
	local RunningDeltaMatrixPart2 = AqwamMatrixLibrary:multiply((1 - self.DecayRate), GradientSquaredMatrix)
	
	local CurrentRunningGradientSquaredMatrix =  AqwamMatrixLibrary:add(RunningDeltaMatrixPart1, RunningDeltaMatrixPart2)
	
	local RootMeanSquarePart1 = AqwamMatrixLibrary:add(CurrentRunningGradientSquaredMatrix, self.Epsilon)
	
	local RootMeanSquare = AqwamMatrixLibrary:applyFunction(math.sqrt, RootMeanSquarePart1)
	
	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, RootMeanSquare)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

	self.PreviousRunningGradientSquaredMatrix = CurrentRunningGradientSquaredMatrix

	return costFunctionDerivatives

end

function AdaptiveGradientDeltaOptimizer:reset()

	self.PreviousRunningGradientSquaredMatrix = nil

end

return AdaptiveGradientDeltaOptimizer
