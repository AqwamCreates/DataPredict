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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseOptimizer = require("Optimizer_BaseOptimizer")

AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

setmetatable(AdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewAdaptiveMomentEstimationOptimizer:setName("AdaptiveMomentEstimation")
	
	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)
	
	NewAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousMomentumTensor = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local previousVelocityTensor = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local momentumTensorPart1 = AqwamTensorLibrary:multiply(NewAdaptiveMomentEstimationOptimizer.beta1, previousMomentumTensor)

		local momentumTensorPart2 = AqwamTensorLibrary:multiply((1 - NewAdaptiveMomentEstimationOptimizer.beta1), costFunctionDerivativeTensor)

		local momentumTensor = AqwamTensorLibrary:add(momentumTensorPart1, momentumTensorPart2)

		local squaredCostFunctionDerivativeTensor = AqwamTensorLibrary:power(costFunctionDerivativeTensor, 2)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(NewAdaptiveMomentEstimationOptimizer.beta2, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - NewAdaptiveMomentEstimationOptimizer.beta2), squaredCostFunctionDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		local meanMomentumTensor = AqwamTensorLibrary:divide(momentumTensor, (1 - NewAdaptiveMomentEstimationOptimizer.beta1))

		local meanVelocityTensor = AqwamTensorLibrary:divide(velocityTensor, (1 - NewAdaptiveMomentEstimationOptimizer.beta2))

		local squareRootedDivisor = AqwamTensorLibrary:power(meanVelocityTensor, 0.5)

		local finalDivisorTensor = AqwamTensorLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(meanMomentumTensor, finalDivisorTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)
		
		NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {momentumTensor, velocityTensor}

		return costFunctionDerivativeTensor
		
	end)

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationOptimizer