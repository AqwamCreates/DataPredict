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

local BaseOptimizer = require("Optimizer_BaseOptimizer")

AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

setmetatable(AdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function AdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewAdaptiveMomentEstimationOptimizer:setName("AdaptiveMomentEstimation")
	
	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)
	
	NewAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local previousMomentumMatrix = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)

		local previousVelocityMatrix = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local timeValue = NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[3] or 1
		
		local beta1 = NewAdaptiveMomentEstimationOptimizer.beta1

		local beta2 = NewAdaptiveMomentEstimationOptimizer.beta2
		
		local weightDecayRate = NewAdaptiveMomentEstimationOptimizer.weightDecayRate

		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end
		
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

		local finalDivisorMatrix = AqwamTensorLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:divide(meanMomentumMatrix, finalDivisorMatrix)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart1)
		
		timeValue = timeValue + 1
		
		NewAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {momentumMatrix, velocityMatrix, timeValue}

		return costFunctionDerivativeMatrix
		
	end)

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationOptimizer
