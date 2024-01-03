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

NesterovAcceleratedAdaptiveMomentEstimationOptimizer = {}

NesterovAcceleratedAdaptiveMomentEstimationOptimizer.__index = NesterovAcceleratedAdaptiveMomentEstimationOptimizer

setmetatable(NesterovAcceleratedAdaptiveMomentEstimationOptimizer, BaseOptimizer)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function NesterovAcceleratedAdaptiveMomentEstimationOptimizer.new(beta1, beta2, epsilon)

	local NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer = BaseOptimizer.new("NesterovAcceleratedAdaptiveMomentEstimation")

	setmetatable(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer, NesterovAcceleratedAdaptiveMomentEstimationOptimizer)

	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousM = nil
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousN = nil
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1 = beta1 or defaultBeta1
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2 = beta2 or defaultBeta2
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon = epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousM = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousM or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousN = NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousN or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local meanCostFunctionDerivatives = AqwamMatrixLibrary:divide(costFunctionDerivatives, (1 - NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1))

		local MPart1 = AqwamMatrixLibrary:multiply(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousM)

		local MPart2 = AqwamMatrixLibrary:multiply((1 - NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1), costFunctionDerivatives)

		local M = AqwamMatrixLibrary:add(MPart1, MPart2)

		local meanM = AqwamMatrixLibrary:divide(M, (1 - NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1))

		local squaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local NPart1 = AqwamMatrixLibrary:multiply(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousN)

		local NPart2 = AqwamMatrixLibrary:multiply((1 - NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2), squaredModelParameters)

		local N = AqwamMatrixLibrary:add(NPart1, NPart2)

		local meanN = AqwamMatrixLibrary:divide(N, (1 - NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta2))

		local finalMPart1 = AqwamMatrixLibrary:multiply((1 - NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1), meanCostFunctionDerivatives)

		local finalMPart2 = AqwamMatrixLibrary:multiply(NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.beta1, meanM)

		local finalM = AqwamMatrixLibrary:add(finalMPart1, finalMPart2)

		local squareRootedDivisor = AqwamMatrixLibrary:power(meanN, 0.5)

		local finalDivisor = AqwamMatrixLibrary:add(squareRootedDivisor, NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.epsilon)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(finalM, finalDivisor)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousM = M

		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousN = N

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer:setResetFunction(function()
		
		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousM = nil

		NewNesterovAcceleratedAdaptiveMomentEstimationOptimizer.previousN = nil
		
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
