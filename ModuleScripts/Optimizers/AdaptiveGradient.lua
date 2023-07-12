AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = {}
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix = nil
	
	return NewAdaptiveGradientOptimizer
	
end

function AdaptiveGradientOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	self.PreviousSumOfGradientSquaredMatrix  = self.PreviousSumOfGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

	local GradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

	local CurrentSumOfGradientSquaredMatrix = AqwamMatrixLibrary:add(self.PreviousSumOfGradientSquaredMatrix, GradientSquaredMatrix)

	local SquareRootSumOfGradientSquared = AqwamMatrixLibrary:power(CurrentSumOfGradientSquaredMatrix, 0.5)

	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, SquareRootSumOfGradientSquared)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)
	
	self.PreviousSumOfGradientSquaredMatrix = CurrentSumOfGradientSquaredMatrix

	return costFunctionDerivatives
	
end

function AdaptiveGradientOptimizer:reset()
	
	self.PreviousSumOfGradientSquaredMatrix = nil
	
end

return AdaptiveGradientOptimizer
