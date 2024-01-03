local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

LearningRateDecayOptimizer = {}

LearningRateDecayOptimizer.__index = LearningRateDecayOptimizer

setmetatable(LearningRateDecayOptimizer, BaseOptimizer)

local defaultDecayRate = 0.5

local defaultTimeStepToDecay = 100

function LearningRateDecayOptimizer.new(decayRate, timeStepToDecay)
	
	local NewLearningRateDecayOptimizer = BaseOptimizer.new("LearningRateDecay")
	
	setmetatable(NewLearningRateDecayOptimizer, LearningRateDecayOptimizer)
	
	NewLearningRateDecayOptimizer.decayRate = decayRate or defaultDecayRate
	
	NewLearningRateDecayOptimizer.timeStepToDecay = timeStepToDecay or defaultTimeStepToDecay
	
	NewLearningRateDecayOptimizer.currentLearningRate = nil
	
	NewLearningRateDecayOptimizer.currentTimeStep = 0
	
	--------------------------------------------------------------------------------
	
	NewLearningRateDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewLearningRateDecayOptimizer.currentTimeStep += 1
		
		local currentLearningRate = NewLearningRateDecayOptimizer.currentLearningRate or learningRate
		
		if ((NewLearningRateDecayOptimizer.currentTimeStep % NewLearningRateDecayOptimizer.timeStepToDecay) == 0) then
			
			currentLearningRate *= NewLearningRateDecayOptimizer.decayRate
			
		end
		
		costFunctionDerivatives = AqwamMatrixLibrary:multiply(currentLearningRate, costFunctionDerivatives)
		
		NewLearningRateDecayOptimizer.currentLearningRate = currentLearningRate

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewLearningRateDecayOptimizer:setResetFunction(function()
		
		NewLearningRateDecayOptimizer.currentLearningRate = nil
		
		NewLearningRateDecayOptimizer.currentTimeStep = 0
		
	end)
	
	return NewLearningRateDecayOptimizer
	
end

function LearningRateDecayOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

function LearningRateDecayOptimizer:setTimeStepToDecay(timeStepToDecay)
	
	self.timeStepToDecay = timeStepToDecay
	
end

return LearningRateDecayOptimizer
