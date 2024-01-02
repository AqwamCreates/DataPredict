local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new("AdaptiveGradient")
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = nil
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local GradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local CurrentSumOfGradientSquaredMatrix = AqwamMatrixLibrary:add(NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix, GradientSquaredMatrix)

		local SquareRootSumOfGradientSquared = AqwamMatrixLibrary:power(CurrentSumOfGradientSquaredMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, SquareRootSumOfGradientSquared)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = CurrentSumOfGradientSquaredMatrix

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setResetFunction(function()
		
		NewAdaptiveGradientOptimizer.previousSumOfGradientSquaredMatrix = nil
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

return AdaptiveGradientOptimizer
