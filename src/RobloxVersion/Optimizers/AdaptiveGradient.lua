--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseOptimizer = require(script.Parent.BaseOptimizer)

AdaptiveGradientOptimizer = {}

AdaptiveGradientOptimizer.__index = AdaptiveGradientOptimizer

setmetatable(AdaptiveGradientOptimizer, BaseOptimizer)

local defaultWeightDecayRate = 0

function AdaptiveGradientOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewAdaptiveGradientOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewAdaptiveGradientOptimizer, AdaptiveGradientOptimizer)
	
	NewAdaptiveGradientOptimizer:setName("AdaptiveGradient")
	
	NewAdaptiveGradientOptimizer.weightDecayRate = NewAdaptiveGradientOptimizer.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveGradientOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewAdaptiveGradientOptimizer.optimizerInternalParameterArray or {}
		
		local previousSumOfGradientSquaredMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local weightDecayRate = NewAdaptiveGradientOptimizer.weightDecayRate
		
		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end
		
		local gradientSquaredMatrix = AqwamTensorLibrary:power(gradientMatrix, 2)

		local currentSumOfGradientSquaredMatrix = AqwamTensorLibrary:add(previousSumOfGradientSquaredMatrix, gradientSquaredMatrix)

		local squareRootSumOfGradientSquaredMatrix = AqwamTensorLibrary:applyFunction(math.sqrt, currentSumOfGradientSquaredMatrix)

		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:divide(gradientMatrix, squareRootSumOfGradientSquaredMatrix)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart1)

		NewAdaptiveGradientOptimizer.optimizerInternalParameterArray = {currentSumOfGradientSquaredMatrix}

		return costFunctionDerivativeMatrix
		
	end)
	
	return NewAdaptiveGradientOptimizer
	
end

function AdaptiveGradientOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

return AdaptiveGradientOptimizer
