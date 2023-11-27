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

AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

local defaultEpsilon = 1 * math.pow(10, -7)

function AdaptiveMomentEstimationOptimizer.new(Beta1, Beta2, Epsilon)

	local NewAdaptiveMomentEstimationOptimizer = {}

	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)

	NewAdaptiveMomentEstimationOptimizer.PreviousMomentum = nil
	
	NewAdaptiveMomentEstimationOptimizer.PreviousVelocity = nil
	
	NewAdaptiveMomentEstimationOptimizer.Beta1 = Beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.Beta2 = Beta2 or defaultBeta2
	
	NewAdaptiveMomentEstimationOptimizer.Epsilon = Epsilon or defaultEpsilon

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(Beta1)
	
	self.Beta1 = Beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(Beta2)
		
	self.Beta2 = Beta2
	
end

function AdaptiveMomentEstimationOptimizer:setEpsilon(Epsilon)

	self.Epsilon = Epsilon

end

function AdaptiveMomentEstimationOptimizer:calculate(learningRate, costFunctionDerivatives)

	self.PreviousMomentum = self.PreviousMomentum or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	self.PreviousVelocity = self.PreviousVelocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

	local momentumPart1 = AqwamMatrixLibrary:multiply(self.Beta1, self.PreviousMomentum)
	
	local momentumPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta1), costFunctionDerivatives)
	
	local momentum = AqwamMatrixLibrary:add(momentumPart1, momentumPart2)

	local squaredModelParameters = AqwamMatrixLibrary:power(costFunctionDerivatives, 2)

	local velocityPart1 = AqwamMatrixLibrary:multiply(self.Beta2, self.PreviousVelocity)
	
	local velocityPart2 = AqwamMatrixLibrary:multiply((1 - self.Beta2), squaredModelParameters)
	
	local velocity = AqwamMatrixLibrary:add(velocityPart1, velocityPart2)

	local meanMomentum = AqwamMatrixLibrary:divide(momentum, (1 - self.Beta1))
	
	local meanVelocity = AqwamMatrixLibrary:divide(velocity, (1 - self.Beta2))
	
	local squareRootedDivisor = AqwamMatrixLibrary:power(meanVelocity, 0.5)
	
	local finalDivisor = AqwamMatrixLibrary:add(squareRootedDivisor, self.Epsilon)

	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:divide(meanMomentum, finalDivisor)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativesPart1)

	self.PreviousMomentum = momentum
	
	self.PreviousVelocity = velocity

	return costFunctionDerivatives
	
end

function AdaptiveMomentEstimationOptimizer:reset()

	self.PreviousMomentum = nil

	self.PreviousVelocity = nil

end

return AdaptiveMomentEstimationOptimizer
