MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultDecayRate = 0.1

function MomentumOptimizer.new(DecayRate)
	
	local NewMomentumOptimizer = {}
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer.DecayRate = DecayRate or defaultDecayRate
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(DecayRate)
	
	self.DecayRate = DecayRate
	
end

function MomentumOptimizer:calculate(costFunctionDerivatives, previousCostFunctionDerivatives)
	
	if (previousCostFunctionDerivatives == nil) then return costFunctionDerivatives end
	
	local MomentumMatrixPart1 = AqwamMatrixLibrary:multiply(self.DecayRate, previousCostFunctionDerivatives)
	
	local costFunctionDerivatives = AqwamMatrixLibrary:add(costFunctionDerivatives, MomentumMatrixPart1)
	
	return costFunctionDerivatives
	
end

function MomentumOptimizer:reset()
	
end

return MomentumOptimizer
