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

AdaptiveDeltaOptimizer = {}

AdaptiveDeltaOptimizer.__index = AdaptiveDeltaOptimizer

setmetatable(AdaptiveDeltaOptimizer, BaseOptimizer)

local defaultDecayRate = 0.9

local defaultWeightDecayRate = 0

local defaultEpsilon = 1e-16

function AdaptiveDeltaOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAdaptiveDeltaOptimizer = BaseOptimizer.new(parameterDictionary)

	setmetatable(NewAdaptiveDeltaOptimizer, AdaptiveDeltaOptimizer)
	
	NewAdaptiveDeltaOptimizer:setName("AdaptiveDelta")
	
	NewAdaptiveDeltaOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	NewAdaptiveDeltaOptimizer.weightDecayRate = NewAdaptiveDeltaOptimizer.weightDecayRate or defaultWeightDecayRate
	
	NewAdaptiveDeltaOptimizer.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	--------------------------------------------------------------------------------
	
	NewAdaptiveDeltaOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewAdaptiveDeltaOptimizer.optimizerInternalParameterArray or {}
		
		local previousRunningGradientSquaredMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local decayRate = NewAdaptiveDeltaOptimizer.decayRate
		
		local weightDecayRate = NewAdaptiveDeltaOptimizer.weightDecayRate

		local gradientMatrix = costFunctionDerivativeMatrix

		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end

		local gradientSquaredMatrix = AqwamTensorLibrary:power(gradientMatrix, 2)

		local runningDeltaMatrixPart1 = AqwamTensorLibrary:multiply(decayRate, previousRunningGradientSquaredMatrix)

		local runningDeltaMatrixPart2 = AqwamTensorLibrary:multiply((1 - decayRate), gradientSquaredMatrix)

		local currentRunningGradientSquaredMatrix = AqwamTensorLibrary:add(runningDeltaMatrixPart1, runningDeltaMatrixPart2)

		local rootMeanSquareMatrixPart1 = AqwamTensorLibrary:add(currentRunningGradientSquaredMatrix, NewAdaptiveDeltaOptimizer.epsilon)

		local rootMeanSquareMatrix = AqwamTensorLibrary:applyFunction(math.sqrt, rootMeanSquareMatrixPart1)

		local costFunctionDerivativeMatrixPart1 = AqwamTensorLibrary:divide(gradientMatrix, rootMeanSquareMatrix)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrixPart1)

		NewAdaptiveDeltaOptimizer.optimizerInternalParameterArray = {currentRunningGradientSquaredMatrix}

		return costFunctionDerivativeMatrix
		
	end)

	return NewAdaptiveDeltaOptimizer

end

function AdaptiveDeltaOptimizer:setDecayRate(decayRate)

	self.decayRate = decayRate

end

function AdaptiveDeltaOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

function AdaptiveDeltaOptimizer:setEpsilon(epsilon)

	self.epsilon = epsilon

end

return AdaptiveDeltaOptimizer
