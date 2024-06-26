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
		
		local currentLearningRate = NewLearningRateTimeDecayOptimizer.optimizerInternalParameters[1] or learningRate

		local currentTimeStep = NewLearningRateTimeDecayOptimizer.optimizerInternalParameters[2] or 0

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
