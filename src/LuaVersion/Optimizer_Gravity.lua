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
	
	--------------------------------------------------------------------------------
	
	NewGravityOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local previousVelocity = NewGravityOptimizer.optimizerInternalParameters[1]
		
		local currentTimeStep = NewGravityOptimizer.optimizerInternalParameters[2] or 0
		
		currentTimeStep += 1
		
		if (previousVelocity == nil) then

			local standardDeviation = NewGravityOptimizer.initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			previousVelocity = AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1], gaussianDensity)

		end

		local meanMovingAverage = ((NewGravityOptimizer.movingAverage * currentTimeStep) + 1) / (currentTimeStep + 2)

		local absoluteM = AqwamMatrixLibrary:applyFunction(math.abs, costFunctionDerivatives)

		local maxM = AqwamMatrixLibrary:findMaximumValue(absoluteM)

		local m = AqwamMatrixLibrary:divide(1, maxM)

		local weirdLPart1 = AqwamMatrixLibrary:divide(costFunctionDerivatives, m)

		local weirdLPart2 = AqwamMatrixLibrary:power(weirdLPart1, 2)

		local weirdLPart3 = AqwamMatrixLibrary:add(1, weirdLPart2)

		local weirdL = AqwamMatrixLibrary:divide(costFunctionDerivatives, weirdLPart3)

		local velocityPart1 = AqwamMatrixLibrary:multiply(meanMovingAverage, previousVelocity)

		local velocityPart2 = AqwamMatrixLibrary:multiply((1 - meanMovingAverage), weirdL)

		local velocity = AqwamMatrixLibrary:add(velocityPart1, velocityPart2)

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(learningRate, velocity) 

		NewGravityOptimizer.optimizerInternalParameters = {velocity, currentTimeStep}

		return costFunctionDerivatives
		
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
