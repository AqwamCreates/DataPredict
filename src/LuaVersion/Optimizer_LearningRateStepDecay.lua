local BaseOptimizer = require("Optimizer_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

LearningRateStepDecayOptimizer = {}

LearningRateStepDecayOptimizer.__index = LearningRateStepDecayOptimizer

setmetatable(LearningRateStepDecayOptimizer, BaseOptimizer)

local defaultDecayRate = 0.5

local defaultTimeStepToDecay = 100

function LearningRateStepDecayOptimizer.new(decayRate, timeStepToDecay)
	
	local NewLearningRateStepDecayOptimizer = BaseOptimizer.new("LearningRateStepDecay")
	
	setmetatable(NewLearningRateStepDecayOptimizer, LearningRateStepDecayOptimizer)
	
	NewLearningRateStepDecayOptimizer.decayRate = decayRate or defaultDecayRate
	
	NewLearningRateStepDecayOptimizer.timeStepToDecay = timeStepToDecay or defaultTimeStepToDecay
	
	NewLearningRateStepDecayOptimizer.currentLearningRate = nil
	
	NewLearningRateStepDecayOptimizer.currentTimeStep = 0
	
	--------------------------------------------------------------------------------
	
	NewLearningRateStepDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		NewLearningRateStepDecayOptimizer.currentTimeStep += 1
		
		local currentLearningRate = NewLearningRateStepDecayOptimizer.currentLearningRate or learningRate
		
		if ((NewLearningRateStepDecayOptimizer.currentTimeStep % NewLearningRateStepDecayOptimizer.timeStepToDecay) == 0) then
			
			currentLearningRate *= NewLearningRateStepDecayOptimizer.decayRate
			
		end
		
		costFunctionDerivatives = AqwamMatrixLibrary:multiply(currentLearningRate, costFunctionDerivatives)
		
		NewLearningRateStepDecayOptimizer.currentLearningRate = currentLearningRate

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewLearningRateStepDecayOptimizer:setResetFunction(function()
		
		NewLearningRateStepDecayOptimizer.currentLearningRate = nil
		
		NewLearningRateStepDecayOptimizer.currentTimeStep = 0
		
	end)
	
	return NewLearningRateStepDecayOptimizer
	
end

function LearningRateStepDecayOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

function LearningRateStepDecayOptimizer:setTimeStepToDecay(timeStepToDecay)
	
	self.timeStepToDecay = timeStepToDecay
	
end

return LearningRateStepDecayOptimizer
