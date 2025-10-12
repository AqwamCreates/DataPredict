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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseOptimizer = require(script.Parent.BaseOptimizer)

AdaptiveMomentEstimationWeightDecayOptimizer = {}

AdaptiveMomentEstimationWeightDecayOptimizer.__index = AdaptiveMomentEstimationWeightDecayOptimizer

setmetatable(AdaptiveMomentEstimationWeightDecayOptimizer, BaseOptimizer)

local defaultAlpha = 0.001

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0.01

local defaultEpsilon = 1e-16

function AdaptiveMomentEstimationWeightDecayOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationWeightDecayOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer:setName("AdaptiveMomentEstimationWeightDecay")
	
	setmetatable(NewAdaptiveMomentEstimationWeightDecayOptimizer, AdaptiveMomentEstimationWeightDecayOptimizer)
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.alpha = parameterDictionary.alpha or defaultBeta1
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationWeightDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local previousMomentumMatrix = NewAdaptiveMomentEstimationWeightDecayOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)

		local previousVelocityMatrix = NewAdaptiveMomentEstimationWeightDecayOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local timeValue = NewAdaptiveMomentEstimationWeightDecayOptimizer.optimizerInternalParameterArray[3] or 1
		
		local beta1 = NewAdaptiveMomentEstimationWeightDecayOptimizer.beta1

		local beta2 = NewAdaptiveMomentEstimationWeightDecayOptimizer.beta2

		local decayedWeightMatrix = AqwamTensorLibrary:multiply(NewAdaptiveMomentEstimationWeightDecayOptimizer.weightDecayRate, weightMatrix)

		local gradientMatrix = AqwamTensorLibrary:add(costFunctionDerivativeMatrix, decayedWeightMatrix)
		
		local momentumMatrixPart1 = AqwamTensorLibrary:multiply(beta1, previousMomentumMatrix)

		local momentumMatrixPart2 = AqwamTensorLibrary:multiply((1 - beta1), gradientMatrix)

		local momentumMatrix = AqwamTensorLibrary:add(momentumMatrixPart1, momentumMatrixPart2)

		local squaredGradientDerivativeMatrix = AqwamTensorLibrary:power(gradientMatrix, 2)

		local velocityMatrixPart1 = AqwamTensorLibrary:multiply(beta2, previousVelocityMatrix)

		local velocityMatrixPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeMatrix)

		local velocityMatrix = AqwamTensorLibrary:add(velocityMatrixPart1, velocityMatrixPart2)

		local meanMomentumMatrix = AqwamTensorLibrary:divide(momentumMatrix, (1 - math.pow(beta1, timeValue)))

		local meanVelocityMatrix = AqwamTensorLibrary:divide(velocityMatrix, (1 - math.pow(beta2, timeValue)))

		local squareRootedDivisor = AqwamTensorLibrary:applyFunction(math.sqrt, meanVelocityMatrix)

		local finalDivisorMatrix = AqwamTensorLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationWeightDecayOptimizer.epsilon)
		
		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:multiply(NewAdaptiveMomentEstimationWeightDecayOptimizer.alpha, meanMomentumMatrix)

		local costFunctionDerivativeMatrixPart2 = AqwamTensorLibrary:divide(costFunctionDerivativeMatrixPart1, finalDivisorMatrix)
		
		local costFunctionDerivativeMatrixPart3 = AqwamTensorLibrary:add(costFunctionDerivativeMatrixPart2, decayedWeightMatrix)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart3)
		
		timeValue = timeValue + 1
		
		NewAdaptiveMomentEstimationWeightDecayOptimizer.optimizerInternalParameterArray = {momentumMatrix, velocityMatrix, timeValue}

		return costFunctionDerivativeMatrix
		
	end)

	return NewAdaptiveMomentEstimationWeightDecayOptimizer

end

function AdaptiveMomentEstimationWeightDecayOptimizer:setAlpha(alpha)

	self.alpha = alpha

end

function AdaptiveMomentEstimationWeightDecayOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationWeightDecayOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationWeightDecayOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveMomentEstimationWeightDecayOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationWeightDecayOptimizer
