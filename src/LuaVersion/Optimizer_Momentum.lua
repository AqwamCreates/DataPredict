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

local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

setmetatable(MomentumOptimizer, BaseOptimizer)

local defaultDecayRate = 0.1

function MomentumOptimizer.new(decayRate)
	
	local NewMomentumOptimizer = BaseOptimizer.new("Momentum")
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer.decayRate = decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local previousVelocity = NewMomentumOptimizer.optimizerInternalParameters[1] or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local velocityPart1 = AqwamMatrixLibrary:multiply(NewMomentumOptimizer.decayRate, previousVelocity)

		local velocityPart2 = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivatives)

		local velocity = AqwamMatrixLibrary:add(velocityPart1, velocityPart2)

		costFunctionDerivatives = velocity
		
		NewMomentumOptimizer.optimizerInternalParameters = {velocity}

		return costFunctionDerivatives
		
	end)
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return MomentumOptimizer
