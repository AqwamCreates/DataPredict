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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseOptimizer = require(script.Parent.BaseOptimizer)

AdaptiveFactorOptimizer = {}

AdaptiveFactorOptimizer.__index = AdaptiveFactorOptimizer

setmetatable(AdaptiveFactorOptimizer, BaseOptimizer)

local defaultBeta2DecayRate = -0.8

local defaultClipValue = 1

local defaultEpsilon1 = 1e-16

local defaultEpsilon2 = 1e-16

function AdaptiveFactorOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveFactorOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveFactorOptimizer, AdaptiveFactorOptimizer)
	
	NewAdaptiveFactorOptimizer:setName("AdaptiveFactor")
	
	NewAdaptiveFactorOptimizer.beta2DecayRate = parameterDictionary.beta2DecayRate or defaultBeta2DecayRate
	
	NewAdaptiveFactorOptimizer.clipValue = parameterDictionary.clipValue or defaultClipValue
	
	NewAdaptiveFactorOptimizer.epsilon1 = parameterDictionary.epsilon1 or defaultEpsilon1
	
	NewAdaptiveFactorOptimizer.epsilon2 = parameterDictionary.epsilon2 or defaultEpsilon2
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveFactorOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor, weightTensor)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor)
		
		local secondMomentRowFactorTensor = NewAdaptiveFactorOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
		
		local secondMomentColumnFactorTensor = NewAdaptiveFactorOptimizer.optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
		
		local timeValue = NewAdaptiveFactorOptimizer.optimizerInternalParameterArray[3] or 1
		
		local beta2DecayRate = NewAdaptiveFactorOptimizer.beta2DecayRate
		
		local beta2 = 1 - math.pow(timeValue, beta2DecayRate)
		
		local oneMinusBeta2 = 1 - beta2
		
		local gradientTensor = costFunctionDerivativeTensor
		
		local squaredGradientTensor = AqwamTensorLibrary:power(gradientTensor, 2)
		
		local oneRowTensor = AqwamTensorLibrary:createTensor({dimensionSizeArray[1], 1}, 1)
		
		local oneColumnTensor = AqwamTensorLibrary:createTensor({dimensionSizeArray[2], 1}, 1)
		
		local transposedOneRowTensor = AqwamTensorLibrary:transpose(oneRowTensor)
		
		local transposedOneColumnTensor = AqwamTensorLibrary:transpose(oneColumnTensor)
		
		local dotProductOnTensor = AqwamTensorLibrary:dotProduct(oneRowTensor, transposedOneColumnTensor)
		
		local epsilonMultiplyDotProductOnTensor = AqwamTensorLibrary:multiply(NewAdaptiveFactorOptimizer.epsilon1, dotProductOnTensor)
		
		local squaredGradientAddEpsilonMultiplyDotProductOnTensor = AqwamTensorLibrary:add(squaredGradientTensor, epsilonMultiplyDotProductOnTensor)
		
		local secondMomentRowFactorTensorPart1 = AqwamTensorLibrary:multiply(beta2, secondMomentRowFactorTensor)
		
		local secondMomentRowFactorTensorPart2 = AqwamTensorLibrary:multiply(oneMinusBeta2, squaredGradientAddEpsilonMultiplyDotProductOnTensor)
		
		local secondMomentRowFactorTensorPart3 = AqwamTensorLibrary:dotProduct(secondMomentRowFactorTensorPart2, oneColumnTensor)
		
		secondMomentRowFactorTensor = AqwamTensorLibrary:add(secondMomentRowFactorTensorPart1, secondMomentRowFactorTensorPart3)
		
		local secondMomentColumnFactorTensorPart1 = AqwamTensorLibrary:multiply(beta2, secondMomentColumnFactorTensor)
		
		local secondMomentColumnFactorTensorPart2 = AqwamTensorLibrary:dotProduct(transposedOneRowTensor, squaredGradientAddEpsilonMultiplyDotProductOnTensor)
		
		local secondMomentColumnFactorTensorPart3 = AqwamTensorLibrary:multiply(oneMinusBeta2, secondMomentRowFactorTensorPart2)
		
		secondMomentColumnFactorTensor = AqwamTensorLibrary:add(secondMomentColumnFactorTensorPart1, secondMomentColumnFactorTensorPart3)
		
		local velocityTensorPart1 = AqwamTensorLibrary:multiply(secondMomentRowFactorTensor, secondMomentColumnFactorTensor)
		
		local velocityTensorPart2 = AqwamTensorLibrary:dotProduct(transposedOneRowTensor, secondMomentRowFactorTensor)
		
		local velocityTensor = AqwamTensorLibrary:divide(velocityTensorPart1, velocityTensorPart2)
		
		local uTensor = AqwamTensorLibrary:divide(gradientTensor, AqwamTensorLibrary:applyFunction(math.sqrt, velocityTensor))
		
		local squareRootVelocityTensor = AqwamTensorLibrary:applyFunction(math.sqrt, velocityTensor)
		
		local dividedRootMeanSquaredXTensor = AqwamTensorLibrary:divide(uTensor, weightTensor)
		
		local momentum = math.min(learningRate, (1 / math.sqrt(timeValue)))
		
		local alpha = AqwamTensorLibrary:applyFunction(math.max, {{NewAdaptiveFactorOptimizer.epsilon2}}, dividedRootMeanSquaredXTensor)
		
		alpha = AqwamTensorLibrary:multiply(alpha, momentum)
		
		local rootMeanSquaredUTensorPart1 = AqwamTensorLibrary:divide(gradientTensor, squareRootVelocityTensor)
		
		local rootMeanSquaredUTensor = AqwamTensorLibrary:unaryMinus(rootMeanSquaredUTensorPart1)
		
		local dividedRootMeanSquaredUTensor = AqwamTensorLibrary:divide(rootMeanSquaredUTensor, NewAdaptiveFactorOptimizer.clipValue)
		
		local finalUTensor = AqwamTensorLibrary:divide(uTensor, AqwamTensorLibrary:applyFunction(math.max, dividedRootMeanSquaredUTensor, {{1}}))
		
		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(learningRate, finalUTensor)
		
		timeValue = timeValue + 1

		NewAdaptiveFactorOptimizer.optimizerInternalParameterArray = {secondMomentRowFactorTensor, secondMomentColumnFactorTensor, timeValue}

		return costFunctionDerivativeTensor
		
	end)

	return NewAdaptiveFactorOptimizer

end

function AdaptiveFactorOptimizer:setBeta2DecayRate(beta2DecayRate)

	self.beta2DecayRate = beta2DecayRate

end

function AdaptiveFactorOptimizer:setClipValue(clipValue)

	self.clipValue = clipValue

end

function AdaptiveFactorOptimizer:setEpsilon1(epsilon1)

	self.epsilon1 = epsilon1

end

function AdaptiveFactorOptimizer:setEpsilon2(epsilon2)

	self.epsilon2 = epsilon2

end

return AdaptiveFactorOptimizer
