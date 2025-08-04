--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseOptimizer = require(script.Parent.BaseOptimizer)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

LearningRateTimeDecayOptimizer = {}

LearningRateTimeDecayOptimizer.__index = LearningRateTimeDecayOptimizer

setmetatable(LearningRateTimeDecayOptimizer, BaseOptimizer)

local defaultDecayRate = 0.5

function LearningRateTimeDecayOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewLearningRateTimeDecayOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewLearningRateTimeDecayOptimizer, LearningRateTimeDecayOptimizer)
	
	NewLearningRateTimeDecayOptimizer:setName("LearningRateDecay")
	
	NewLearningRateTimeDecayOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewLearningRateTimeDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local currentLearningRate = NewLearningRateTimeDecayOptimizer.optimizerInternalParameterArray[1] or learningRate

		local currentTimeStep = NewLearningRateTimeDecayOptimizer.optimizerInternalParameterArray[2] or 0

		currentTimeStep += 1
			
		currentLearningRate = currentLearningRate / (NewLearningRateTimeDecayOptimizer.decayRate * currentTimeStep)
		
		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(currentLearningRate, costFunctionDerivativeTensor)
		
		NewLearningRateTimeDecayOptimizer.optimizerInternalParameterArray = {currentLearningRate, currentTimeStep}

		return costFunctionDerivativeTensor
		
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
