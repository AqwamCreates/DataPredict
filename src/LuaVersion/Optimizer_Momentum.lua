--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Neural)

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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseOptimizer = require("Optimizer_BaseOptimizer")

MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

setmetatable(MomentumOptimizer, BaseOptimizer)

local defaultDecayRate = 0.1

function MomentumOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMomentumOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer:setName("Momentum")
	
	NewMomentumOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local previousVelocityTensor = NewMomentumOptimizer.optimizerInternalParameterArray[1] or AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(costFunctionDerivativeTensor), 0)

		local velocityTensorPart1 = AqwamTensorLibrary:multiply(NewMomentumOptimizer.decayRate, previousVelocityTensor)

		local velocityTensorPart2 = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeTensor)

		local velocityTensor = AqwamTensorLibrary:add(velocityTensorPart1, velocityTensorPart2)

		costFunctionDerivativeTensor = velocityTensor
		
		NewMomentumOptimizer.optimizerInternalParameterArray = {velocityTensor}

		return costFunctionDerivativeTensor
		
	end)
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return MomentumOptimizer