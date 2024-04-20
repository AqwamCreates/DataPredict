local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

LearningRateTimeDecayOptimizer = {}

LearningRateTimeDecayOptimizer.__index = LearningRateTimeDecayOptimizer

setmetatable(LearningRateTimeDecayOptimizer, BaseOptimizer)

local defaultDecayRate = 0.5

local defaultTimeStepToDecay = 100

function LearningRateTimeDecayOptimizer.new(decayRate, timeStepToDecay)
	
	local NewLearningRateTimeDecayOptimizer = BaseOptimizer.new("LearningRateDecay")
	
	setmetatable(NewLearningRateTimeDecayOptimizer, LearningRateTimeDecayOptimizer)
	
	NewLearningRateTimeDecayOptimizer.decayRate = decayRate or defaultDecayRate
	
	NewLearningRateTimeDecayOptimizer.timeStepToDecay = timeStepToDecay or defaultTimeStepToDecay
	
	NewLearningRateTimeDecayOptimizer.currentLearningRate = nil
	
	NewLearningRateTimeDecayOptimizer.currentTimeStep = 0
	
	--------------------------------------------------------------------------------
	
	NewLearningRateTimeDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewLearningRateTimeDecayOptimizer.currentTimeStep = NewLearningRateTimeDecayOptimizer.currentTimeStep + 1
		
		local currentLearningRate = NewLearningRateTimeDecayOptimizer.currentLearningRate or learningRate
			
		currentLearningRate /= (NewLearningRateTimeDecayOptimizer.decayRate * NewLearningRateTimeDecayOptimizer.currentTimeStep)
		
		costFunctionDerivatives = AqwamMatrixLibrary:multiply(currentLearningRate, costFunctionDerivatives)
		
		NewLearningRateTimeDecayOptimizer.currentLearningRate = currentLearningRate

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewLearningRateTimeDecayOptimizer:setResetFunction(function()
		
		NewLearningRateTimeDecayOptimizer.currentLearningRate = nil
		
		NewLearningRateTimeDecayOptimizer.currentTimeStep = 0
		
	end)
	
	return NewLearningRateTimeDecayOptimizer
	
end

function LearningRateTimeDecayOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

function LearningRateTimeDecayOptimizer:setTimeStepToDecay(timeStepToDecay)
	
	self.timeStepToDecay = timeStepToDecay
	
end

return LearningRateTimeDecayOptimizer
