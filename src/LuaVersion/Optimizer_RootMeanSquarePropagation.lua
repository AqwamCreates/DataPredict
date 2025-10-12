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

RootMeanSquarePropagationOptimizer = {}

RootMeanSquarePropagationOptimizer.__index = RootMeanSquarePropagationOptimizer

setmetatable(RootMeanSquarePropagationOptimizer, BaseOptimizer)

local defaultBeta = 0.1

local defaultWeightDecayRate = 0

local defaultEpsilonValue = 1e-16

function RootMeanSquarePropagationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewRootMeanSquarePropagationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewRootMeanSquarePropagationOptimizer, RootMeanSquarePropagationOptimizer)
	
	NewRootMeanSquarePropagationOptimizer:setName("RootMeanSquarePropagation")
	
	NewRootMeanSquarePropagationOptimizer.beta = parameterDictionary.beta or defaultBeta
	
	NewRootMeanSquarePropagationOptimizer.weightDecayRate = NewRootMeanSquarePropagationOptimizer.weightDecayRate or defaultWeightDecayRate
	
	NewRootMeanSquarePropagationOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilonValue
	
	--------------------------------------------------------------------------------
	
	NewRootMeanSquarePropagationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local previousVelocity = NewRootMeanSquarePropagationOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local beta = NewRootMeanSquarePropagationOptimizer.beta
		
		local weightDecayRate = NewRootMeanSquarePropagationOptimizer.weightDecayRate
		
		local gradientMatrix = costFunctionDerivativeMatrix

		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end

		local squaredCostFunctionDerivativeMatrix = AqwamTensorLibrary:power(gradientMatrix, 2)

		local vMatrixPart1 = AqwamTensorLibrary:multiply(beta, previousVelocity)

		local vMatrixPart2 = AqwamTensorLibrary:multiply((1 - beta), squaredCostFunctionDerivativeMatrix)

		local velocityMatrix = AqwamTensorLibrary:add(vMatrixPart1, vMatrixPart2)

		local velocityNonZeroDivisorMatrix = AqwamTensorLibrary:add(velocityMatrix, NewRootMeanSquarePropagationOptimizer.epsilon)

		local squaredRootVelocityMatrix = AqwamTensorLibrary:applyFunction(math.sqrt, velocityNonZeroDivisorMatrix)

		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:divide(gradientMatrix, squaredRootVelocityMatrix)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart1)

		NewRootMeanSquarePropagationOptimizer.optimizerInternalParameterArray = {velocityMatrix}

		return costFunctionDerivativeMatrix
		
	end)
	
	return NewRootMeanSquarePropagationOptimizer
	
end

function RootMeanSquarePropagationOptimizer:setBeta(beta)
	
	self.beta = beta
	
end

function RootMeanSquarePropagationOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function RootMeanSquarePropagationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return RootMeanSquarePropagationOptimizer
