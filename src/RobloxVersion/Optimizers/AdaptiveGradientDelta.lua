--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

AdaptiveGradientDeltaOptimizer = {}

AdaptiveGradientDeltaOptimizer.__index = AdaptiveGradientDeltaOptimizer

setmetatable(AdaptiveGradientDeltaOptimizer, BaseOptimizer)

local defaultDecayRate = 0.9

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveGradientDeltaOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveGradientDeltaOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveGradientDeltaOptimizer, AdaptiveGradientDeltaOptimizer)
	
	NewAdaptiveGradientDeltaOptimizer:setName("AdaptiveGradientDelta")
	
	NewAdaptiveGradientDeltaOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	NewAdaptiveGradientDeltaOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientDeltaOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousRunningGradientSquaredTensor = NewAdaptiveGradientDeltaOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local decayRate = NewAdaptiveGradientDeltaOptimizer.decayRate

		local gradientSquaredTensor = AqwamTensorLibrary:power(costFunctionDerivativeTensor, 2)

		local runningDeltaTensorPart1 = AqwamTensorLibrary:multiply(decayRate, previousRunningGradientSquaredTensor)

		local runningDeltaTensorPart2 = AqwamTensorLibrary:multiply((1 - decayRate), gradientSquaredTensor)

		local currentRunningGradientSquaredTensor =  AqwamTensorLibrary:add(runningDeltaTensorPart1, runningDeltaTensorPart2)

		local rootMeanSquareTensorPart1 = AqwamTensorLibrary:add(currentRunningGradientSquaredTensor, NewAdaptiveGradientDeltaOptimizer.epsilon)

		local rootMeanSquareTensor = AqwamTensorLibrary:applyFunction(math.sqrt, rootMeanSquareTensorPart1)

		local costFunctionDerivativesPart1 = AqwamTensorLibrary:divide(costFunctionDerivativeTensor, rootMeanSquareTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveGradientDeltaOptimizer.optimizerInternalParameterArray = {currentRunningGradientSquaredTensor}

		return costFunctionDerivativeTensor
		
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
