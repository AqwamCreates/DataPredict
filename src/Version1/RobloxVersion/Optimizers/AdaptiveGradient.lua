local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new("AdaptiveGradient")
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local previousSumOfGradientSquaredMatrix = NewAdaptiveGradientOptimizer.optimizerInternalParameters or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local gradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local currentSumOfGradientSquaredMatrix = AqwamMatrixLibrary:add(previousSumOfGradientSquaredMatrix, gradientSquaredMatrix)

		local squareRootSumOfGradientSquared = AqwamMatrixLibrary:power(currentSumOfGradientSquaredMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, squareRootSumOfGradientSquared)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientOptimizer.optimizerInternalParameters = currentSumOfGradientSquaredMatrix

		return costFunctionDerivatives
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

return AdaptiveGradientOptimizer
