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

AdaptiveMomentEstimationMaximumOptimizer = {}

AdaptiveMomentEstimationMaximumOptimizer.__index = AdaptiveMomentEstimationMaximumOptimizer

setmetatable(AdaptiveMomentEstimationMaximumOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function AdaptiveMomentEstimationMaximumOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationMaximumOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveMomentEstimationMaximumOptimizer, AdaptiveMomentEstimationMaximumOptimizer)
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setName("AdaptiveMomentEstimationMaximum")

	NewAdaptiveMomentEstimationMaximumOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationMaximumOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationMaximumOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveMomentEstimationMaximumOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationMaximumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameterArray or {}

		local momentMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)

		local exponentWeightMatrix = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1
		
		local beta1 = NewAdaptiveMomentEstimationMaximumOptimizer.beta1
		
		local beta2 = NewAdaptiveMomentEstimationMaximumOptimizer.beta2
		
		local weightDecayRate = NewAdaptiveMomentEstimationMaximumOptimizer.weightDecayRate
		
		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then
			
			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)
			
			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)
			
		end

		local momentMatrixPart1 = AqwamTensorLibrary:multiply(beta1, momentMatrix)

		local momentMatrixPart2 = AqwamTensorLibrary:multiply((1 - beta1), gradientMatrix)

		momentMatrix = AqwamTensorLibrary:add(momentMatrixPart1, momentMatrixPart2)

		local exponentWeightMatrixPart1 = AqwamTensorLibrary:multiply(beta2, exponentWeightMatrix)

		local exponentWeightMatrixPart2 = AqwamTensorLibrary:applyFunction(math.abs, gradientMatrix)

		exponentWeightMatrix = AqwamTensorLibrary:applyFunction(math.max, exponentWeightMatrixPart1, exponentWeightMatrixPart2)

		local divisorMatrixPart1 = 1 - math.pow(beta1, timeValue)

		local divisorMatrixPart2 = AqwamTensorLibrary:add(exponentWeightMatrix, NewAdaptiveMomentEstimationMaximumOptimizer.epsilon)

		local divisorMatrix = AqwamTensorLibrary:multiply(divisorMatrixPart1, divisorMatrixPart2)

		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:divide(momentMatrix, divisorMatrix)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart1)
		
		NewAdaptiveMomentEstimationMaximumOptimizer.optimizerInternalParameterArray = {momentMatrix, exponentWeightMatrix, timeValue}

		return costFunctionDerivativeMatrix

	end)

	return NewAdaptiveMomentEstimationMaximumOptimizer

end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationMaximumOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveMomentEstimationMaximumOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationMaximumOptimizer
