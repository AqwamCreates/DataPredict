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

local MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

setmetatable(MomentumOptimizer, BaseOptimizer)

local defaultDecayRate = 0.1

local defaultWeightDecayRate = 0

function MomentumOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMomentumOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer:setName("Momentum")
	
	NewMomentumOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	NewMomentumOptimizer.weightDecayRate = NewMomentumOptimizer.weightDecayRate or defaultWeightDecayRate
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeMatrix, weightMatrix)
		
		local optimizerInternalParameterArray = NewMomentumOptimizer.optimizerInternalParameterArray or {}
		
		local previousVelocityMatrix = optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeMatrix), 0)
		
		local weightDecayRate = NewMomentumOptimizer.weightDecayRate

		local gradientMatrix = costFunctionDerivativeMatrix

		if (weightDecayRate ~= 0) then

			local decayedWeightMatrix = AqwamTensorLibrary:multiply(weightDecayRate, weightMatrix)

			gradientMatrix = AqwamTensorLibrary:add(gradientMatrix, decayedWeightMatrix)

		end
		
		local velocityMatrixPart1 = AqwamTensorLibrary:multiply(NewMomentumOptimizer.decayRate, previousVelocityMatrix)

		local velocityMatrixPart2 = AqwamTensorLibrary:multiply(learningRate, gradientMatrix)

		local velocityMatrix = AqwamTensorLibrary:add(velocityMatrixPart1, velocityMatrixPart2)

		costFunctionDerivativeMatrix = velocityMatrix
		
		NewMomentumOptimizer.optimizerInternalParameterArray = {velocityMatrix}

		return costFunctionDerivativeMatrix
		
	end)
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

function MomentumOptimizer:setWeightDecayRate(weightDecayRate)

	self.weightDecayRate = weightDecayRate

end

return MomentumOptimizer
