--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

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

function GravityOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGravityOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewGravityOptimizer, GravityOptimizer)
	
	NewGravityOptimizer:setName("Gravity")
	
	NewGravityOptimizer.initialStepSize = parameterDictionary.initialStepSize or defaultInitialStepSize
	
	NewGravityOptimizer.movingAverage = parameterDictionary.movingAverage or defaultMovingAverage
	
	--------------------------------------------------------------------------------
	
	NewGravityOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousVelocityTensor = NewGravityOptimizer.optimizerInternalParameterArray[1]
		
		local currentTimeStep = NewGravityOptimizer.optimizerInternalParameterArray[2] or 0
		
		currentTimeStep += 1
		
		if (previousVelocityTensor == nil) then

			local standardDeviation = NewGravityOptimizer.initialStepSize / learningRate

			local gaussianDensity = calculateGaussianDensity(0, standardDeviation)

			previousVelocityTensor = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), gaussianDensity)

		end

		local meanMovingAverage = ((NewGravityOptimizer.movingAverage * currentTimeStep) + 1) / (currentTimeStep + 2)

		local absoluteMTensor = AqwamTensorLibrary:applyFunction(math.abs, costFunctionDerivativeTensor)

		local maxMTensor = AqwamTensorLibrary:findMaximumValue(absoluteMTensor)

		local mTensor = AqwamTensorLibrary:divide(1, maxMTensor)

		local weirdLTensorPart1 = AqwamTensorLibrary:divide(costFunctionDerivativeTensor, mTensor)

		local weirdLTensorPart2 = AqwamTensorLibrary:power(weirdLTensorPart1, 2)

		local weirdLTensorPart3 = AqwamTensorLibrary:add(1, weirdLTensorPart2)

		local weirdLTensor = AqwamTensorLibrary:divide(costFunctionDerivativeTensor, weirdLTensorPart3)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(meanMovingAverage, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply((1 - meanMovingAverage), weirdLTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, velocityTensor) 

		NewGravityOptimizer.optimizerInternalParameterArray = {velocityTensor, currentTimeStep}

		return costFunctionDerivativeTensor
		
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
