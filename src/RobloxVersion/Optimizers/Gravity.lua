local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

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
	
	--------------------------------------------------------------------------------
	
	NewLearningRateStepDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local currentLearningRate = NewLearningRateStepDecayOptimizer.optimizerInternalParameters[1] or learningRate
		
		local currentTimeStep = NewLearningRateStepDecayOptimizer.optimizerInternalParameters[2] or 0

		currentTimeStep += 1
		
		if ((currentTimeStep % NewLearningRateStepDecayOptimizer.timeStepToDecay) == 0) then
			
			currentLearningRate *= NewLearningRateStepDecayOptimizer.decayRate
			
		end
		
		costFunctionDerivatives = AqwamMatrixLibrary:multiply(currentLearningRate, costFunctionDerivatives)
		
		NewLearningRateStepDecayOptimizer.optimizerInternalParameters = {currentLearningRate, currentTimeStep}

		return costFunctionDerivatives
		
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
