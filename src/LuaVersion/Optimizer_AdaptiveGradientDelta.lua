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

local BaseOptimizer = require("BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

AdaptiveGradientDeltaOptimizer = {}

AdaptiveGradientDeltaOptimizer.__index = AdaptiveGradientDeltaOptimizer

setmetatable(AdaptiveGradientDeltaOptimizer, BaseOptimizer)

local defaultDecayRate = 0.9

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveGradientDeltaOptimizer.new(decayRate, epsilon)

	local NewAdaptiveGradientDeltaOptimizer = BaseOptimizer.new("AdaptiveGradientDelta")

	setmetatable(NewAdaptiveGradientDeltaOptimizer, AdaptiveGradientDeltaOptimizer)

	NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = nil
	
	NewAdaptiveGradientDeltaOptimizer.decayRate = decayRate or defaultDecayRate
	
	NewAdaptiveGradientDeltaOptimizer.epsilon = epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientDeltaOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local GradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local RunningDeltaMatrixPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveGradientDeltaOptimizer.decayRate, NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix)

		local RunningDeltaMatrixPart2 = AqwamMatrixLibrary:multiply((1 - NewAdaptiveGradientDeltaOptimizer.decayRate), GradientSquaredMatrix)

		local CurrentRunningGradientSquaredMatrix =  AqwamMatrixLibrary:add(RunningDeltaMatrixPart1, RunningDeltaMatrixPart2)

		local RootMeanSquarePart1 = AqwamMatrixLibrary:add(CurrentRunningGradientSquaredMatrix, NewAdaptiveGradientDeltaOptimizer.epsilon)

		local RootMeanSquare = AqwamMatrixLibrary:applyFunction(math.sqrt, RootMeanSquarePart1)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, RootMeanSquare)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = CurrentRunningGradientSquaredMatrix

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientDeltaOptimizer:setResetFunction(function()
		
		NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = nil
		
	end)

	return NewAdaptiveGradientDeltaOptimizer

end

function AdaptiveGradientDeltaOptimizer:setEpsilon(Epsilon)

	self.Epsilon = Epsilon

end

function AdaptiveGradientDeltaOptimizer:setDecayRate(DecayRate)
	
	self.DecayRate = DecayRate
	
end

return AdaptiveGradientDeltaOptimizer
