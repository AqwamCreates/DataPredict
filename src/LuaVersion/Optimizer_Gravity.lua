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

GravityOptimizer = {}

GravityOptimizer.__index = GravityOptimizer

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultInitialStepSize = 0.01

local defaultMovingAverage = 0.9

function GravityOptimizer.new(InitialStepSize, MovingAverage)
	
	local NewGravityOptimizer = {}
	
	setmetatable(NewGravityOptimizer, GravityOptimizer)
	
	NewGravityOptimizer.InitialStepSize = InitialStepSize or defaultInitialStepSize
	
	NewGravityOptimizer.MovingAverage = MovingAverage or defaultMovingAverage
	
	NewGravityOptimizer.PreviousVelocity = nil
	
	NewGravityOptimizer.TimeStep = 0
	
	return NewGravityOptimizer
	
end

function GravityOptimizer:setInitialStepSize(InitialStepSize)
	
	self.InitialStepSize = InitialStepSize
	
end

function GravityOptimizer:setMovingAverage(MovingAverage)

	self.MovingAverage = MovingAverage

end

local function calculateGaussianDensity(mean, standardDeviation)

	local exponentStep1 = math.pow(mean, 2)

	local exponentPart2 = math.pow(standardDeviation, 2)

	local exponentStep3 = exponentStep1 / exponentPart2

	local exponentStep4 = -0.5 * exponentStep3

	local exponentWithTerms = math.exp(exponentStep4)

	local divisor = standardDeviation * math.sqrt(2 * math.pi)

	local gaussianDensity = exponentWithTerms / divisor

	return gaussianDensity

end

function GravityOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	if (self.PreviousVelocity == nil) then
		
		local standardDeviation = self.InitialStepSize / learningRate
		
		local gaussianDensity = calculateGaussianDensity(0, standardDeviation)
		
		self.PreviousVelocity = AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1], gaussianDensity)
		
	end
	
	self.TimeStep += 1
	
	local meanMovingAverage = ((self.MovingAverage * self.TimeStep) + 1) / (self.TimeStep + 2)
	
	local AbsoluteM = AqwamMatrixLibrary:applyFunction(math.abs, costFunctionDerivatives)
	
	local maxM = AqwamMatrixLibrary:findMaximumValueInMatrix(AbsoluteM)
	
	local M = AqwamMatrixLibrary:divide(1, maxM)
	
	local WeirdLPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, M)
	
	local WeirdLPart2 = AqwamMatrixLibrary:power(WeirdLPart1, 2)
	
	local WeirdLPart3 = AqwamMatrixLibrary:add(1, WeirdLPart2)
	
	local WeirdL = AqwamMatrixLibrary:divide(costFunctionDerivatives, WeirdLPart3)
	
	local VelocityPart1 = AqwamMatrixLibrary:multiply(meanMovingAverage, self.PreviousVelocity)
	
	local VelocityPart2 = AqwamMatrixLibrary:multiply((1 - meanMovingAverage), WeirdL)
	
	local Velocity = AqwamMatrixLibrary:add(VelocityPart1, VelocityPart2)
	
	costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, Velocity) 
	
	self.PreviousVelocity = Velocity
	
	return costFunctionDerivatives
	
end

function GravityOptimizer:reset()
	
	self.PreviousVelocity = nil
	
	self.TimeStep = 0
	
end

return GravityOptimizer
