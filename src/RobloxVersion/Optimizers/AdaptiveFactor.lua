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

local defaultWeightDecayRate = 0

local defaultClipValue = 1

local defaultEpsilon1 = 1e-16

local defaultEpsilon2 = 1e-16

function AdaptiveFactorOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveFactorOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveFactorOptimizer, AdaptiveFactorOptimizer)
	
	NewAdaptiveFactorOptimizer:setName("AdaptiveFactor")
	
	NewAdaptiveFactorOptimizer.beta2DecayRate = parameterDictionary.beta2DecayRate or defaultBeta2DecayRate
	
	NewAdaptiveFactorOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveFactorOptimizer.clipValue = parameterDictionary.clipValue or defaultClipValue
	
	NewAdaptiveFactorOptimizer.epsilon1 = parameterDictionary.epsilon1 or defaultEpsilon1
	
	NewAdaptiveFactorOptimizer.epsilon2 = parameterDictionary.epsilon2 or defaultEpsilon2
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveFactorOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix)
		
		local optimizerInternalParameterArray = NewAdaptiveFactorOptimizer.optimizerInternalParameterArray or {}
		
		local secondMomentRowFactorMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
		
		local secondMomentColumnFactorMatrix = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)
		
		local timeValue = (optimizerInternalParameterArray[3] or 0) + 1
		
		local beta2DecayRate = NewAdaptiveFactorOptimizer.beta2DecayRate
		
		local weightDecayRate = NewAdaptiveFactorOptimizer.weightDecayRate
		
		local beta2 = 1 - math.pow(timeValue, beta2DecayRate)
		
		local oneMinusBeta2 = 1 - beta2
		
		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end
		
		local squaredGradientMatrix = AqwamTensorLibrary:power(gradientMatrix, 2)
		
		local oneRowMatrix = AqwamTensorLibrary:createTensor({dimensionSizeArray[1], 1}, 1)
		
		local oneColumnMatrix = AqwamTensorLibrary:createTensor({dimensionSizeArray[2], 1}, 1)
		
		local transposedOneRowMatrix = AqwamTensorLibrary:transpose(oneRowMatrix)
		
		local transposedOneColumnMatrix = AqwamTensorLibrary:transpose(oneColumnMatrix)
		
		local dotProductOnMatrix = AqwamTensorLibrary:dotProduct(oneRowMatrix, transposedOneColumnMatrix)
		
		local epsilonMultiplyDotProductOnMatrix = AqwamTensorLibrary:multiply(NewAdaptiveFactorOptimizer.epsilon1, dotProductOnMatrix)
		
		local squaredGradientAddEpsilonMultiplyDotProductOnMatrix = AqwamTensorLibrary:add(squaredGradientMatrix, epsilonMultiplyDotProductOnMatrix)
		
		local secondMomentRowFactorMatrixPart1 = AqwamTensorLibrary:multiply(beta2, secondMomentRowFactorMatrix)
		
		local secondMomentRowFactorMatrixPart2 = AqwamTensorLibrary:multiply(oneMinusBeta2, squaredGradientAddEpsilonMultiplyDotProductOnMatrix)
		
		local secondMomentRowFactorMatrixPart3 = AqwamTensorLibrary:dotProduct(secondMomentRowFactorMatrixPart2, oneColumnMatrix)
		
		secondMomentRowFactorMatrix = AqwamTensorLibrary:add(secondMomentRowFactorMatrixPart1, secondMomentRowFactorMatrixPart3)
		
		local secondMomentColumnFactorMatrixPart1 = AqwamTensorLibrary:multiply(beta2, secondMomentColumnFactorMatrix)
		
		local secondMomentColumnFactorMatrixPart2 = AqwamTensorLibrary:dotProduct(transposedOneRowMatrix, squaredGradientAddEpsilonMultiplyDotProductOnMatrix)
		
		local secondMomentColumnFactorMatrixPart3 = AqwamTensorLibrary:multiply(oneMinusBeta2, secondMomentRowFactorMatrixPart2)
		
		secondMomentColumnFactorMatrix = AqwamTensorLibrary:add(secondMomentColumnFactorMatrixPart1, secondMomentColumnFactorMatrixPart3)
		
		local velocityMatrixPart1 = AqwamTensorLibrary:multiply(secondMomentRowFactorMatrix, secondMomentColumnFactorMatrix)
		
		local velocityMatrixPart2 = AqwamTensorLibrary:dotProduct(transposedOneRowMatrix, secondMomentRowFactorMatrix)
		
		local velocityMatrix = AqwamTensorLibrary:divide(velocityMatrixPart1, velocityMatrixPart2)
		
		local uMatrix = AqwamTensorLibrary:divide(gradientMatrix, AqwamTensorLibrary:applyFunction(math.sqrt, velocityMatrix))
		
		local squareRootVelocityMatrix = AqwamTensorLibrary:applyFunction(math.sqrt, velocityMatrix)
		
		local dividedRootMeanSquaredXMatrix = AqwamTensorLibrary:divide(uMatrix, weightMatrix)
		
		local momentum = math.min(learningRate, (1 / math.sqrt(timeValue)))
		
		local alpha = AqwamTensorLibrary:applyFunction(math.max, {{NewAdaptiveFactorOptimizer.epsilon2}}, dividedRootMeanSquaredXMatrix)
		
		alpha = AqwamTensorLibrary:multiply(alpha, momentum)
		
		local rootMeanSquaredUMatrixPart1 = AqwamTensorLibrary:divide(gradientMatrix, squareRootVelocityMatrix)
		
		local rootMeanSquaredUMatrix = AqwamTensorLibrary:unaryMinus(rootMeanSquaredUMatrixPart1)
		
		local dividedRootMeanSquaredUMatrix = AqwamTensorLibrary:divide(rootMeanSquaredUMatrix, NewAdaptiveFactorOptimizer.clipValue)
		
		local finalUMatrix = AqwamTensorLibrary:divide(uMatrix, AqwamTensorLibrary:applyFunction(math.max, dividedRootMeanSquaredUMatrix, {{1}}))
		
		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, finalUMatrix)

		NewAdaptiveFactorOptimizer.optimizerInternalParameterArray = {secondMomentRowFactorMatrix, secondMomentColumnFactorMatrix, timeValue}

		return costFunctionDerivativeMatrix
		
	end)

	return NewAdaptiveFactorOptimizer

end

function AdaptiveFactorOptimizer:setBeta2DecayRate(beta2DecayRate)

	self.beta2DecayRate = beta2DecayRate

end

function AdaptiveFactorOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

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
