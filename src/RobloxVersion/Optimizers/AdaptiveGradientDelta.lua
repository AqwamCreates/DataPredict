local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

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
	
	NewAdaptiveGradientDeltaOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		local decayRate = NewAdaptiveGradientDeltaOptimizer.decayRate

		local gradientSquaredMatrix = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local runningDeltaMatrixPart1 = AqwamMatrixLibrary:multiply(decayRate, NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix)

		local runningDeltaMatrixPart2 = AqwamMatrixLibrary:multiply((1 - decayRate), gradientSquaredMatrix)

		local currentRunningGradientSquaredMatrix =  AqwamMatrixLibrary:add(runningDeltaMatrixPart1, runningDeltaMatrixPart2)

		local rootMeanSquarePart1 = AqwamMatrixLibrary:add(currentRunningGradientSquaredMatrix, NewAdaptiveGradientDeltaOptimizer.epsilon)

		local rootMeanSquare = AqwamMatrixLibrary:applyFunction(math.sqrt, rootMeanSquarePart1)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, rootMeanSquare)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = currentRunningGradientSquaredMatrix

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientDeltaOptimizer:setResetFunction(function()
		
		NewAdaptiveGradientDeltaOptimizer.previousRunningGradientSquaredMatrix = nil
		
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
