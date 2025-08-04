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

AdaptiveMomentEstimationMaximumOptimizer = {}

AdaptiveMomentEstimationMaximumOptimizer.__index = AdaptiveMomentEstimationMaximumOptimizer

setmetatable(AdaptiveMomentEstimationMaximumOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationMaximumOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationMaximumOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveMomentEstimationMaximumOptimizer, AdaptiveMomentEstimationMaximumOptimizer)
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setName("AdaptiveMomentEstimationMaximum")

	NewAdaptiveMomentEstimationMaximumOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationMaximumOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationMaximumOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)

		local momentTensor = NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local exponentWeightTensor = NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local beta1 = NewAdaptiveMomentEstimationMaximumOptimizer.beta1
		
		local beta2 = NewAdaptiveMomentEstimationMaximumOptimizer.beta2

		local momentTensorPart1 = AqwamTensorLibrary:multiply(beta1, momentTensor)

		local momentTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), costFunctionDerivativeTensor)

		momentTensor = AqwamTensorLibrary:add(momentTensorPart1, momentTensorPart2)

		local exponentWeightTensorPart1 = AqwamTensorLibrary:multiply(beta2, exponentWeightTensor)

		local exponentWeightTensorPart2 = AqwamTensorLibrary:applyFunction(math.abs, costFunctionDerivativeTensor)

		exponentWeightTensor = AqwamTensorLibrary:applyFunction(math.max, exponentWeightTensorPart1, exponentWeightTensorPart2)

		local divisorTensorPart1 = 1 - math.pow(beta1, 2)

		local divisorTensorPart2 = AqwamTensorLibrary:add(exponentWeightTensor, NewAdaptiveMomentEstimationMaximumOptimizer.epsilon)

		local divisorTensor = AqwamTensorLibrary:multiply(divisorTensorPart1, divisorTensorPart2)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(momentTensor, divisorTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)
		
		NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameterArray = {momentTensor, exponentWeightTensor}

		return costFunctionDerivativeTensor

	end)

	return NewAdaptiveMomentEstimationMaximumOptimizer

end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationMaximumOptimizer
