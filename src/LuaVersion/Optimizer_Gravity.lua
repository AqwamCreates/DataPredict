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

GravityOptimizer = {}

GravityOptimizer.__index = GravityOptimizer

setmetatable(GravityOptimizer, BaseOptimizer)

local defaultInitialStepSize = 0.01

local defaultMovingAverage = 0.9

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

function GravityOptimizer.new(initialStepSize, movingAverage)
	
	local NewGravityOptimizer = BaseOptimizer.new("Gravity")
	
	setmetatable(NewGravityOptimizer, GravityOptimizer)
	
	NewGravityOptimizer.initialStepSize = initialStepSize or defaultInitialStepSize
	
	NewGravityOptimizer.movingAverage = movingAverage or defaultMovingAverage
	
	NewGravityOptimizer.previousVelocity = nil
	
	NewGravityOptimizer.timeStep = 0
	
	--------------------------------------------------------------------------------
	
	NewGravityOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		if (NewGravityOptimizer.previousVelocity == nil) then

			local standardDeviation = NewGravityOptimizer.initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			NewGravityOptimizer.previousVelocity = AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1], gaussianDensity)

		end

		NewGravityOptimizer.timeStep += 1

		local meanMovingAverage = ((NewGravityOptimizer.movingAverage * NewGravityOptimizer.timeStep) + 1) / (NewGravityOptimizer.timeStep + 2)

		local AbsoluteM = AqwamMatrixLibrary:applyFunction(math.abs, costFunctionDerivatives)

		local maxM = AqwamMatrixLibrary:findMaximumValueInMatrix(AbsoluteM)

		local M = AqwamMatrixLibrary:divide(1, maxM)

		local WeirdLPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, M)

		local WeirdLPart2 = AqwamMatrixLibrary:power(WeirdLPart1, 2)

		local WeirdLPart3 = AqwamMatrixLibrary:add(1, WeirdLPart2)

		local WeirdL = AqwamMatrixLibrary:divide(costFunctionDerivatives, WeirdLPart3)

		local VelocityPart1 = AqwamMatrixLibrary:multiply(meanMovingAverage, NewGravityOptimizer.previousVelocity)

		local VelocityPart2 = AqwamMatrixLibrary:multiply((1 - meanMovingAverage), WeirdL)

		local Velocity = AqwamMatrixLibrary:add(VelocityPart1, VelocityPart2)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, Velocity) 

		NewGravityOptimizer.previousVelocity = Velocity

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewGravityOptimizer:setResetFunction(function()
		
		NewGravityOptimizer.previousVelocity = nil

		NewGravityOptimizer.timeStep = 0
		
	end)
	
	return NewGravityOptimizer
	
end

function GravityOptimizer:setInitialStepSize(initialStepSize)
	
	self.initialStepSize = initialStepSize
	
end

function GravityOptimizer:setMovingAverage(movingAverage)

	self.movingAverage = movingAverage

end

return GravityOptimizer
