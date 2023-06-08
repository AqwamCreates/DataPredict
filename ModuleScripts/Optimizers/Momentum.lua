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

function MomentumOptimizer:calculate(ModelParametersDerivatives, PreviousDeltaMatrix)
	
	if (PreviousDeltaMatrix == nil) then
		
		PreviousDeltaMatrix = AqwamMatrixLibrary:createMatrix(#ModelParametersDerivatives, #ModelParametersDerivatives[1])
		
	end
	
	local MomentumMatrixPart1 = AqwamMatrixLibrary:multiply(self.DecayRate, PreviousDeltaMatrix)
	
	local costFunctionDerivatives = AqwamMatrixLibrary:add(ModelParametersDerivatives, MomentumMatrixPart1)
	
	return costFunctionDerivatives
	
end

function MomentumOptimizer:reset()
	
end

return MomentumOptimizer
