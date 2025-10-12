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

GravityOptimizer = {}

GravityOptimizer.__index = GravityOptimizer

setmetatable(GravityOptimizer, BaseOptimizer)

local defaultInitialStepSize = 0.01

local defaultMovingAverage = 0.9

local defaultWeightDecayRate = 0

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

function GravityOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGravityOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewGravityOptimizer, GravityOptimizer)
	
	NewGravityOptimizer:setName("Gravity")
	
	NewGravityOptimizer.initialStepSize = parameterDictionary.initialStepSize or defaultInitialStepSize
	
	NewGravityOptimizer.movingAverage = parameterDictionary.movingAverage or defaultMovingAverage
	
	NewGravityOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewGravityOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local previousVelocityMatrix = NewGravityOptimizer.optimizerInternalParameterArray[1]
		
		local timeValue = NewGravityOptimizer.optimizerInternalParameterArray[2] or 1
		
		local weightDecayRate = NewGravityOptimizer.weightDecayRate
		
		if (not previousVelocityMatrix) then

			local standardDeviation = NewGravityOptimizer.initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			previousVelocityMatrix = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), gaussianDensity)

		end
		
		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end

		local meanMovingAverage = ((NewGravityOptimizer.movingAverage * timeValue) + 1) / (timeValue + 2)

		local absoluteGradientMatrix = AqwamTensorLibrary:applyFunction(math.abs, gradientMatrix)

		local maximumGradientValue = AqwamTensorLibrary:findMaximumValue(absoluteGradientMatrix)

		local mMatrix = AqwamTensorLibrary:divide(1, maximumGradientValue)

		local weirdLMatrixPart1 = AqwamTensorLibrary:divide(gradientMatrix, mMatrix)

		local weirdLMatrixPart2 = AqwamTensorLibrary:power(weirdLMatrixPart1, 2)

		local weirdLMatrixPart3 = AqwamTensorLibrary:add(1, weirdLMatrixPart2)

		local weirdLMatrix = AqwamTensorLibrary:divide(gradientMatrix, weirdLMatrixPart3)

		local velocityMatrixPart1 = AqwamTensorLibrary:multiply(meanMovingAverage, previousVelocityMatrix)

		local velocityMatrixPart2 = AqwamTensorLibrary:multiply((1 - meanMovingAverage), weirdLMatrix)

		local velocityMatrix = AqwamTensorLibrary:add(velocityMatrixPart1, velocityMatrixPart2)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, velocityMatrix)
		
		timeValue = timeValue + 1

		NewGravityOptimizer.optimizerInternalParameterArray = {velocityMatrix, timeValue}

		return costFunctionDerivativeMatrix
		
	end)
	
	return NewGravityOptimizer
	
end

function GravityOptimizer:setInitialStepSize(initialStepSize)
	
	self.initialStepSize = initialStepSize
	
end

function GravityOptimizer:setMovingAverage(movingAverage)

	self.movingAverage = movingAverage

end

function GravityOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

return GravityOptimizer
