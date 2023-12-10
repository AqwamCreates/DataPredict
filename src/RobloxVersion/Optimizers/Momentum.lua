MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultDecayRate = 0.1

function MomentumOptimizer.new(DecayRate)
	
	local NewMomentumOptimizer = {}
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer.DecayRate = DecayRate or defaultDecayRate
	
	NewMomentumOptimizer.Velocity = nil
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(DecayRate)
	
	self.DecayRate = DecayRate
	
end

function MomentumOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	self.Velocity = self.Velocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])
	
	local VelocityPart1 = AqwamMatrixLibrary:multiply(self.DecayRate, self.Velocity)
	
	local VelocityPart2 = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivatives)
	
	self.Velocity = AqwamMatrixLibrary:add(VelocityPart1, VelocityPart2)
	
	costFunctionDerivatives = self.Velocity
	
	return costFunctionDerivatives
	
end

function MomentumOptimizer:reset()
	
	self.Velocity = nil
	
end

return MomentumOptimizer
