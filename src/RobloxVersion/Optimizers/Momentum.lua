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
	
	NewMomentumOptimizer.velocity = nil
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)
		
		NewMomentumOptimizer.velocity = NewMomentumOptimizer.velocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local VelocityPart1 = AqwamMatrixLibrary:multiply(NewMomentumOptimizer.decayRate, NewMomentumOptimizer.velocity)

		local VelocityPart2 = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivatives)

		NewMomentumOptimizer.velocity = AqwamMatrixLibrary:add(VelocityPart1, VelocityPart2)

		costFunctionDerivatives = NewMomentumOptimizer.velocity

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setResetFunction(function()
		
		NewMomentumOptimizer.velocity = nil
		
	end) 
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return MomentumOptimizer
