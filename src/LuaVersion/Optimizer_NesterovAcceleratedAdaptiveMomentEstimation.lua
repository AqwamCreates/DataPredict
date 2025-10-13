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

NesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

NesterovAcceleratedAdaptiveMomentEstimationOptimizer.__index = NesterovAcceleratedAdaptiveMomentEstimationOptimizer

setmetatable(NesterovAcceleratedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer, NesterovAcceleratedAdaptiveMomentEstimationOptimizer)
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setName("NesterovAcceleratedAdaptiveMomentEstimation")
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray or {}
		
		local previousMMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)

		local previousNMatrix = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local timeValue = optimizerInternalParameterArray[3] or 1
		
		local beta1 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1
		
		local beta2 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2
		
		local weightDecayRate = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.weightDecayRate
		
		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end
		
		local oneMinusBeta1 = (1 - beta1)

		local meanCostFunctionDerivativeMatrix = AqwamTensorLibrary:divide(gradientMatrix, oneMinusBeta1)

		local mMatrixPart1 = AqwamTensorLibrary:multiply(beta1, previousMMatrix)

		local mMatrixPart2 = AqwamTensorLibrary:multiply(oneMinusBeta1, gradientMatrix)

		local mMatrix = AqwamTensorLibrary:add(mMatrixPart1, mMatrixPart2)

		local meanMMatrix = AqwamTensorLibrary:divide(mMatrix, oneMinusBeta1)

		local squaredGradientDerivativeMatrix = AqwamTensorLibrary:power(gradientMatrix, 2)

		local nMatrixPart1 = AqwamTensorLibrary:multiply(beta2, previousNMatrix)

		local nMatrixPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredGradientDerivativeMatrix)

		local nMatrix = AqwamTensorLibrary:add(nMatrixPart1, nMatrixPart2)
		
		local multipliedNMatrix = AqwamTensorLibrary:multiply(beta2, nMatrix)

		local meanNMatrix = AqwamTensorLibrary:divide(multipliedNMatrix, (1 - math.pow(beta2, timeValue)))

		local finalMMatrixPart1 = AqwamTensorLibrary:multiply(oneMinusBeta1, meanCostFunctionDerivativeMatrix)

		local finalMMatrixPart2 = AqwamTensorLibrary:multiply(beta1, meanMMatrix)

		local finalMMatrix = AqwamTensorLibrary:add(finalMMatrixPart1, finalMMatrixPart2)

		local squareRootedDivisor = AqwamTensorLibrary:applyFunction(math.sqrt, meanNMatrix)

		local finalDivisor = AqwamTensorLibrary:add(squareRootedDivisor, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:divide(finalMMatrix, finalDivisor)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart1)
		
		timeValue = timeValue + 1
		
		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {mMatrix, nMatrix, timeValue}

		return costFunctionDerivativeMatrix
		
	end)
	
	return NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return NesterovAcceleratedAdaptiveMomentEstimationOptimizer
