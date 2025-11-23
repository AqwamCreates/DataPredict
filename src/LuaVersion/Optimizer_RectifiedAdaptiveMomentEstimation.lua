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

local RectifiedAdaptiveMomentEstimationOptimizer = {}

RectifiedAdaptiveMomentEstimationOptimizer.__index = RectifiedAdaptiveMomentEstimationOptimizer

setmetatable(RectifiedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function RectifiedAdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewRectifiedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	NewRectifiedAdaptiveMomentEstimationOptimizer:setName("RectifiedAdaptiveMomentEstimation")
	
	setmetatable(NewRectifiedAdaptiveMomentEstimationOptimizer, RectifiedAdaptiveMomentEstimationOptimizer)
	
	local beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.beta2 = beta2
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewRectifiedAdaptiveMomentEstimationOptimizer.pInfinity = ((2 / (1 - beta2)) - 1)
	
	--------------------------------------------------------------------------------
	
	NewRectifiedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewRectifiedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray or {}
		
		local previousMomentumMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)

		local previousVelocityMatrix = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1
		
		local beta1 = NewRectifiedAdaptiveMomentEstimationOptimizer.beta1

		local beta2 = NewRectifiedAdaptiveMomentEstimationOptimizer.beta2
		
		local weightDecayRate = NewRectifiedAdaptiveMomentEstimationOptimizer.weightDecayRate
		
		local pInfinity = NewRectifiedAdaptiveMomentEstimationOptimizer.pInfinity

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
		
		local powerBeta2 = math.pow(beta2, timeValue)
		
		local p = pInfinity - ((2 * timeValue * powerBeta2) / (1 - powerBeta2))
		
		if (p > 4) then
			
			local squareRootVelocityMatrix = AqwamTensorLibrary:applyFunction(math.sqrt, velocityMatrix)
			
			local adaptiveLearningRateMatrixPart1 = AqwamTensorLibrary:add(squareRootVelocityMatrix, NewRectifiedAdaptiveMomentEstimationOptimizer.epsilon)
			
			local adaptiveLearningRateMatrix = AqwamTensorLibrary:divide((1 - powerBeta2), adaptiveLearningRateMatrixPart1)
			
			local varianceRectificationNominatorValue = (p - 4) * (p - 2) * pInfinity
			
			local varianceRectificationDenominatorValue = (pInfinity - 4) * (pInfinity - 2) * p
			
			local varianceRectificationValue =  math.sqrt(varianceRectificationNominatorValue / varianceRectificationDenominatorValue)
			
			costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply((learningRate * varianceRectificationValue), meanMomentumMatrix, adaptiveLearningRateMatrix)
			
		else
			
			costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, meanMomentumMatrix)
			
		end
		
		NewRectifiedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {momentumMatrix, velocityMatrix, timeValue}

		return costFunctionDerivativeMatrix
		
	end)

	return NewRectifiedAdaptiveMomentEstimationOptimizer

end

function RectifiedAdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function RectifiedAdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function RectifiedAdaptiveMomentEstimationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function RectifiedAdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return RectifiedAdaptiveMomentEstimationOptimizer
