local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new("AdaptiveGradient")
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix = nil
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix = NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local GradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local CurrentSumOfGradientSquaredMatrix = AqwamMatrixLibrary:add(NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix, GradientSquaredMatrix)

		local SquareRootSumOfGradientSquared = AqwamMatrixLibrary:power(CurrentSumOfGradientSquaredMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, SquareRootSumOfGradientSquared)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix = CurrentSumOfGradientSquaredMatrix

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setResetFunction(function()
		
		NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix = nil
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

return AdaptiveGradientOptimizer
