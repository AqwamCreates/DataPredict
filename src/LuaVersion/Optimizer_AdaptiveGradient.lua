local BaseOptimizer = require("Optimizer_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new("AdaptiveGradient")
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = nil
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local gradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local currentSumOfGradientSquaredMatrix = AqwamMatrixLibrary:add(NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix, gradientSquaredMatrix)

		local squareRootSumOfGradientSquared = AqwamMatrixLibrary:power(currentSumOfGradientSquaredMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, squareRootSumOfGradientSquared)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = currentSumOfGradientSquaredMatrix

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setResetFunction(function()
		
		NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = nil
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

return AdaptiveGradientOptimizer
