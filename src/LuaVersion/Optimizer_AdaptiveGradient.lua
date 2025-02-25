--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Neural)

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

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

function AdaptiveGradientOptimizer.new(parameterDictionary)
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer:setName("AdaptiveGradient")
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousSumOfGradientSquaredTensor = NewAdaptiveGradientOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local gradientSquaredTensor = AqwamTensorLibrary:power(costFunctionDerivativeTensor, 2)

		local currentSumOfGradientSquaredTensor = AqwamTensorLibrary:add(previousSumOfGradientSquaredTensor, gradientSquaredTensor)

		local squareRootSumOfGradientSquaredTensor = AqwamTensorLibrary:power(currentSumOfGradientSquaredTensor, 0.5)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(costFunctionDerivativeTensor, squareRootSumOfGradientSquaredTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)

		NewAdaptiveGradientOptimizer.optimizerInternalParameterArray = {currentSumOfGradientSquaredTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

return AdaptiveGradientOptimizer