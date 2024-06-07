--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

RootMeanSquarePropagationOptimizer = {}

RootMeanSquarePropagationOptimizer.__index = RootMeanSquarePropagationOptimizer

setmetatable(RootMeanSquarePropagationOptimizer, BaseOptimizer)

local defaultBetaValue = 0.1

local defaultEpsilonValue = 1 * math.pow(10, -7)

function RootMeanSquarePropagationOptimizer.new(beta, epsilon)
	
	local NewRootMeanSquarePropagationOptimizer = BaseOptimizer.new("RootMeanSquarePropagation")
	
	setmetatable(NewRootMeanSquarePropagationOptimizer, RootMeanSquarePropagationOptimizer)
	
	NewRootMeanSquarePropagationOptimizer.beta = beta or defaultBetaValue
	
	NewRootMeanSquarePropagationOptimizer.epsilon = epsilon or defaultEpsilonValue
	
	NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix = nil
	
	--------------------------------------------------------------------------------
	
	NewRootMeanSquarePropagationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix = NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
		
		local beta = NewRootMeanSquarePropagationOptimizer.beta

		local squaredCostFunctionDerivatives = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local vMatrixPart1 = AqwamMatrixLibrary:multiply(beta, NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix)

		local vMatrixPart2 = AqwamMatrixLibrary:multiply((1 - beta), squaredCostFunctionDerivatives)

		local currentVelocityMatrix = AqwamMatrixLibrary:add(vMatrixPart1, vMatrixPart2)

		local nonZeroDivisorMatrix = AqwamMatrixLibrary:add(currentVelocityMatrix, NewRootMeanSquarePropagationOptimizer.epsilon)

		local squaredRootVelocityMatrix = AqwamMatrixLibrary:power(nonZeroDivisorMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, squaredRootVelocityMatrix)

		local costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix = currentVelocityMatrix

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewRootMeanSquarePropagationOptimizer:setResetFunction(function()
		
		NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix = nil
		
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
