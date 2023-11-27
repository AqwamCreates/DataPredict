--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]
AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

function AdaptiveGradientOptimizer.new()
	
	local NewAdaptiveGradientOptimizer = {}
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer.PreviousSumOfGradientSquaredMatrix = nil
	
	return NewAdaptiveGradientOptimizer
	
end

function AdaptiveGradientOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	self.PreviousSumOfGradientSquaredMatrix = self.PreviousSumOfGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

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
