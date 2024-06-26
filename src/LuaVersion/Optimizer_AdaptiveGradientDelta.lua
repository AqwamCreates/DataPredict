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

AdaptiveGradientDeltaOptimizer = {}

AdaptiveGradientDeltaOptimizer.__index = AdaptiveGradientDeltaOptimizer

setmetatable(AdaptiveGradientDeltaOptimizer, BaseOptimizer)

local defaultDecayRate = 0.9

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveGradientDeltaOptimizer.new(decayRate, epsilon)

	local NewAdaptiveGradientDeltaOptimizer = BaseOptimizer.new("AdaptiveGradientDelta")

	setmetatable(NewAdaptiveGradientDeltaOptimizer, AdaptiveGradientDeltaOptimizer)
	
	NewAdaptiveGradientDeltaOptimizer.decayRate = decayRate or defaultDecayRate
	
	NewAdaptiveGradientDeltaOptimizer.epsilon = epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientDeltaOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local previousRunningGradientSquaredMatrix = NewAdaptiveGradientDeltaOptimizer.optimizerInternalParameters[1] or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		local decayRate = NewAdaptiveGradientDeltaOptimizer.decayRate

		local gradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local runningDeltaMatrixPart1 = AqwamMatrixLibrary:multiply(decayRate, previousRunningGradientSquaredMatrix)

		local runningDeltaMatrixPart2 = AqwamMatrixLibrary:multiply((1 - decayRate), gradientSquaredMatrix)

		local currentRunningGradientSquaredMatrix =  AqwamMatrixLibrary:add(runningDeltaMatrixPart1, runningDeltaMatrixPart2)

		local rootMeanSquarePart1 = AqwamMatrixLibrary:add(currentRunningGradientSquaredMatrix, NewAdaptiveGradientDeltaOptimizer.epsilon)

		local rootMeanSquare = AqwamMatrixLibrary:applyFunction(math.sqrt, rootMeanSquarePart1)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, rootMeanSquare)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientDeltaOptimizer.optimizerInternalParameters = {currentRunningGradientSquaredMatrix}

		return costFunctionDerivatives
		
	end)

	return NewAdaptiveGradientDeltaOptimizer

end

function AdaptiveGradientDeltaOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

function AdaptiveGradientDeltaOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return AdaptiveGradientDeltaOptimizer
