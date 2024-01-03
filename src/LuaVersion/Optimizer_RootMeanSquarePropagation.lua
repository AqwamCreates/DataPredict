--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseOptimizer = require("Optimizer_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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

		local SquaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local VMatrixPart1 = AqwamMatrixLibrary:multiply(NewRootMeanSquarePropagationOptimizer.beta, NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix)

		local VMatrixPart2 = AqwamMatrixLibrary:multiply((1-NewRootMeanSquarePropagationOptimizer.beta), SquaredModelParameters)

		local CurrentVelocityMatrix = AqwamMatrixLibrary:add(VMatrixPart1, VMatrixPart2)

		local NonZeroDivisorMatrix = AqwamMatrixLibrary:add(CurrentVelocityMatrix, NewRootMeanSquarePropagationOptimizer.epsilon)

		local SquaredRootVelocityMatrix = AqwamMatrixLibrary:power(NonZeroDivisorMatrix, 0.5)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, SquaredRootVelocityMatrix)

		local costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewRootMeanSquarePropagationOptimizer.previousVelocityMatrix = CurrentVelocityMatrix

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
