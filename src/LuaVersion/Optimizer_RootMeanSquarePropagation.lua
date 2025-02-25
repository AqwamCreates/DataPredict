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

RootMeanSquarePropagationOptimizer = {}

RootMeanSquarePropagationOptimizer.__index = RootMeanSquarePropagationOptimizer

setmetatable(RootMeanSquarePropagationOptimizer, BaseOptimizer)

local defaultBetaValue = 0.1

local defaultEpsilonValue = 1 * math.pow(10, -7)

function RootMeanSquarePropagationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewRootMeanSquarePropagationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewRootMeanSquarePropagationOptimizer, RootMeanSquarePropagationOptimizer)
	
	NewRootMeanSquarePropagationOptimizer:setName("RootMeanSquarePropagation")
	
	NewRootMeanSquarePropagationOptimizer.beta = parameterDictionary.beta or defaultBetaValue
	
	NewRootMeanSquarePropagationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilonValue
	
	--------------------------------------------------------------------------------
	
	NewRootMeanSquarePropagationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousVelocity = NewRootMeanSquarePropagationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)
		
		local beta = NewRootMeanSquarePropagationOptimizer.beta

		local squaredCostFunctionDerivativeTensor = AqwamTensorLibrary:power(costFunctionDerivativeTensor, 2)

		local vTensorPart1 = AqwamTensorLibrary:multiply(beta, previousVelocity)

		local vTensorPart2 = AqwamTensorLibrary:multiply((1 - beta), squaredCostFunctionDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(vTensorPart1, vTensorPart2)

		local velocityNonZeroDivisorTensor = AqwamTensorLibrary:add(velocityTensor, NewRootMeanSquarePropagationOptimizer.epsilon)

		local squaredRootVelocityTensor = AqwamTensorLibrary:power(velocityNonZeroDivisorTensor, 0.5)

		local costFunctionDerivativeTensorPart1 = AqwamTensorLibrary:divide(costFunctionDerivativeTensor, squaredRootVelocityTensor)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensorPart1)

		NewRootMeanSquarePropagationOptimizer.optimizerInternalParameterArray = {velocityTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewRootMeanSquarePropagationOptimizer
	
end

function RootMeanSquarePropagationOptimizer:setBeta(beta)
	
	self.beta = beta
	
end

function RootMeanSquarePropagationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return RootMeanSquarePropagationOptimizer