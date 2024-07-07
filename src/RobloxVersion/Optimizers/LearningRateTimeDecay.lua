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
	
	--------------------------------------------------------------------------------
	
	NewLearningRateTimeDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivatives)
		
		local currentLearningRate

		local currentTimeStep

		local optimizerInternalParameters = NewLearningRateTimeDecayOptimizer.optimizerInternalParameters

		if (optimizerInternalParameters) then

			currentLearningRate = optimizerInternalParameters[1]

			currentTimeStep = optimizerInternalParameters[2]

		end
		
		currentLearningRate = currentLearningRate or learningRate

		currentTimeStep = currentTimeStep or 0

		currentTimeStep += 1
			
		currentLearningRate = currentLearningRate / (NewLearningRateTimeDecayOptimizer.decayRate * currentTimeStep)
		
		costFunctionDerivatives = AqwamMatrixLibrary:multiply(currentLearningRate, costFunctionDerivatives)
		
		NewLearningRateTimeDecayOptimizer.optimizerInternalParameters = {currentLearningRate, currentTimeStep}

		return costFunctionDerivatives
		
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
