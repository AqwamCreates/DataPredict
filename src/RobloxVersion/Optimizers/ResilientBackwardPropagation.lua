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

ResilientBackwardPropagationOptimizer = {}

ResilientBackwardPropagationOptimizer.__index = ResilientBackwardPropagationOptimizer

setmetatable(ResilientBackwardPropagationOptimizer, BaseOptimizer)

local defaultEtaPlus = 0.5

local defaultEtaMinus = 1.2

local defaultMinimumStepSize = 1e-6

local defaultMaximumStepSize = 50

local defaultWeightDecayRate = 0

function ResilientBackwardPropagationOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewResilientBackwardPropagationOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewResilientBackwardPropagationOptimizer, ResilientBackwardPropagationOptimizer)
	
	NewResilientBackwardPropagationOptimizer:setName("ResilientBackwardPropagation")
	
	NewResilientBackwardPropagationOptimizer.etaPlus = parameterDictionary.etaPlus or defaultEtaPlus
	
	NewResilientBackwardPropagationOptimizer.etaMinus = parameterDictionary.etaMinus or defaultEtaMinus
	
	NewResilientBackwardPropagationOptimizer.maximumStepSize = parameterDictionary.maximumStepSize or defaultMaximumStepSize
	
	NewResilientBackwardPropagationOptimizer.minimumStepSize = parameterDictionary.minimumStepSize or defaultMinimumStepSize
	
	NewResilientBackwardPropagationOptimizer.weightDecayRate = parameterDictionary.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewResilientBackwardPropagationOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewResilientBackwardPropagationOptimizer.optimizerInternalParameterArray or {}
		
		local previousGradientMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local learningRateMatrix = optimizerInternalParameterArray[2] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), learningRate)
		
		local etaPlus = NewResilientBackwardPropagationOptimizer.etaPlus
		
		local etaMinus = NewResilientBackwardPropagationOptimizer.etaMinus
		
		local maximumStepSize = NewResilientBackwardPropagationOptimizer.maximumStepSize
		
		local minimumStepSize = NewResilientBackwardPropagationOptimizer.minimumStepSize
		
		local weightDecayRate = NewResilientBackwardPropagationOptimizer.weightDecayRate
		
		local gradientMatrix = costFunctionDerivativeMatrix
		
		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end
		
		local multipliedGradientMatrix = AqwamTensorLibrary:multiply(gradientMatrix, previousGradientMatrix)
		
		for i, unwrappedMultipliedGradientVector in ipairs(multipliedGradientMatrix) do
			
			for j, unwrappedGradientValue in ipairs(unwrappedMultipliedGradientVector) do
				
				if (unwrappedGradientValue > 0) then
					
					learningRateMatrix[i][j] = math.min(learningRateMatrix[i][j] * etaPlus, maximumStepSize)
					
				elseif (unwrappedGradientValue < 0) then
					
					learningRateMatrix[i][j] = math.max(learningRateMatrix[i][j] * etaMinus, minimumStepSize)
					
					gradientMatrix[i][j] = 0
					
				end
				
			end
			
		end
		
		local signMatrix = AqwamTensorLibrary:applyFunction(math.sign, gradientMatrix)
		
		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRateMatrix, signMatrix)

		NewResilientBackwardPropagationOptimizer.optimizerInternalParameterArray = {gradientMatrix, learningRateMatrix}

		return costFunctionDerivativeMatrix
		
	end)
	
	return NewResilientBackwardPropagationOptimizer
	
end

function ResilientBackwardPropagationOptimizer:setEtaPlus(etaPlus)
	
	self.etaPlus = etaPlus
	
end

function ResilientBackwardPropagationOptimizer:setEtaMinus(etaMinus)

	self.etaMinus = etaMinus

end

function ResilientBackwardPropagationOptimizer:setMaximumStepSize(maximumStepSize)

	self.maximumStepSize = maximumStepSize

end

function ResilientBackwardPropagationOptimizer:setMinimumStepSize(minimumStepSize)

	self.minimumStepSize = minimumStepSize

end

return ResilientBackwardPropagationOptimizer
