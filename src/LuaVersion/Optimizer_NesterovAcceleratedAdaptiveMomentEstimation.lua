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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseOptimizer = require("Optimizer_BaseOptimizer")

NesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

NesterovAcceleratedAdaptiveMomentEstimationOptimizer.__index = NesterovAcceleratedAdaptiveMomentEstimationOptimizer

setmetatable(NesterovAcceleratedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer, NesterovAcceleratedAdaptiveMomentEstimationOptimizer)
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setName("NesterovAcceleratedAdaptiveMomentEstimation")
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1 = parameterDictionary.beta1 or defaultBeta1

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2 = parameterDictionary.beta2 or defaultBeta2

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousMTensor = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local previousNTensor = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray[2]  or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local beta1 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1
		
		local beta2 = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2

		local meanCostFunctionDerivativeTensor = AqwamTensorLibrary:divide(costFunctionDerivativeTensor, (1 - beta1))

		local mTensorPart1 = AqwamTensorLibrary:multiply(beta1, previousMTensor)

		local mTensorPart2 = AqwamTensorLibrary:multiply((1 - beta1), costFunctionDerivativeTensor)

		local mTensor = AqwamTensorLibrary:add(mTensorPart1, mTensorPart2)

		local meanMTensor = AqwamTensorLibrary:divide(mTensor, (1 - beta1))

		local squaredCostFunctionDerivatives = AqwamTensorLibrary:power(costFunctionDerivativeTensor, 2)

		local nTensorPart1 = AqwamTensorLibrary:multiply(beta2, previousNTensor)

		local nTensorPart2 = AqwamTensorLibrary:multiply((1 - beta2), squaredCostFunctionDerivatives)

		local nTensor = AqwamTensorLibrary:add(nTensorPart1, nTensorPart2)

		local meanNTensor = AqwamTensorLibrary:divide(nTensor, (1 - beta2))

		local finalMTensorPart1 = AqwamTensorLibrary:multiply((1 - beta1), meanCostFunctionDerivativeTensor)

		local finalMTensorPart2 = AqwamTensorLibrary:multiply(beta1, meanMTensor)

		local finalMTensor = AqwamTensorLibrary:add(finalMTensorPart1, finalMTensorPart2)

		local squareRootedDivisor = AqwamTensorLibrary:power(meanNTensor, 0.5)

		local finalDivisor = AqwamTensorLibrary:add(squareRootedDivisor, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativesPart1 = AqwamTensorLibrary:divide(finalMTensor, finalDivisor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.optimizerInternalParameterArray = {mTensor, nTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer

end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return NesterovAcceleratedAdaptiveMomentEstimationOptimizer
