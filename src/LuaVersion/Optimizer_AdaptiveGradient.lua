--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseOptimizer = require("Optimizer_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new("AdaptiveGradient")
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local previousSumOfGradientSquaredMatrix = NewAdaptiveGradientOptimizer.optimizerInternalParameters[1] or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local gradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local currentSumOfGradientSquaredMatrix = AqwamMatrixLibrary:add(previousSumOfGradientSquaredMatrix, gradientSquaredMatrix)

		local squareRootSumOfGradientSquared = AqwamMatrixLibrary:power(currentSumOfGradientSquaredMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, squareRootSumOfGradientSquared)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientOptimizer.optimizerInternalParameters = {currentSumOfGradientSquaredMatrix}

		return costFunctionDerivatives
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

return AdaptiveGradientOptimizer
