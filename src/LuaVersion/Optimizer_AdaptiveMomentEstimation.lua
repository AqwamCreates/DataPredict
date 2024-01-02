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

local BaseModel = require("Optimizer_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

setmetatable(AdaptiveMomentEstimationOptimizer, BaseModel)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationOptimizer.new(beta1, beta2, epsilon)

	local NewAdaptiveMomentEstimationOptimizer = BaseModel.new("AdaptiveMomentEstimation")

	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)

	NewAdaptiveMomentEstimationOptimizer.previousMomentum = nil
	
	NewAdaptiveMomentEstimationOptimizer.previousVelocity = nil
	
	NewAdaptiveMomentEstimationOptimizer.beta1 = beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.beta2 = beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationOptimizer.epsilon = epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)
		
		NewAdaptiveMomentEstimationOptimizer.previousMomentum = NewAdaptiveMomentEstimationOptimizer.previousMomentum or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		NewAdaptiveMomentEstimationOptimizer.previousVelocity = NewAdaptiveMomentEstimationOptimizer.previousVelocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local momentumPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveMomentEstimationOptimizer.beta1, NewAdaptiveMomentEstimationOptimizer.previousMomentum)

		local momentumPart2 = AqwamMatrixLibrary:multiply((1 - NewAdaptiveMomentEstimationOptimizer.beta1), costFunctionDerivatives)

		local momentum = AqwamMatrixLibrary:add(momentumPart1, momentumPart2)

		local squaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

		local velocityPart1 = AqwamMatrixLibrary:multiply(NewAdaptiveMomentEstimationOptimizer.beta2, NewAdaptiveMomentEstimationOptimizer.PreviousVelocity)

		local velocityPart2 = AqwamMatrixLibrary:multiply((1 - NewAdaptiveMomentEstimationOptimizer.beta2), squaredModelParameters)

		local velocity = AqwamMatrixLibrary:add(velocityPart1, velocityPart2)

		local meanMomentum = AqwamMatrixLibrary:divide(momentum, (1 - NewAdaptiveMomentEstimationOptimizer.beta1))

		local meanVelocity = AqwamMatrixLibrary:divide(velocity, (1 - NewAdaptiveMomentEstimationOptimizer.beta2))

		local squareRootedDivisor = AqwamMatrixLibrary:power(meanVelocity, 0.5)

		local finalDivisor = AqwamMatrixLibrary:add(squareRootedDivisor, NewAdaptiveMomentEstimationOptimizer.Epsilon)

		local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(meanMomentum, finalDivisor)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

		NewAdaptiveMomentEstimationOptimizer.previousMomentum = momentum

		NewAdaptiveMomentEstimationOptimizer.previousVelocity = velocity

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveMomentEstimationOptimizer:setResetFunction(function()
		
		NewAdaptiveMomentEstimationOptimizer.previousMomentum = nil

		NewAdaptiveMomentEstimationOptimizer.previousVelocity = nil
		
	end)

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(beta1)
	
	self.beta1 = beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(beta2)
		
	self.beta2 = beta2
	
end

function AdaptiveMomentEstimationOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveMomentEstimationOptimizer
