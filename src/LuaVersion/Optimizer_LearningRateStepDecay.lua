--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Neural)

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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseOptimizer = require("Optimizer_BaseOptimizer")

LearningRateStepDecayOptimizer = {}

LearningRateStepDecayOptimizer.__index = LearningRateStepDecayOptimizer

setmetatable(LearningRateStepDecayOptimizer, BaseOptimizer)

local defaultDecayRate = 0.5

local defaultTimeStepToDecay = 100

function LearningRateStepDecayOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewLearningRateStepDecayOptimizer = BaseOptimizer.new(parameterDictionary)
	
	setmetatable(NewLearningRateStepDecayOptimizer, LearningRateStepDecayOptimizer)
	
	NewLearningRateStepDecayOptimizer:setName("LearningRateStepDecay")
	
	NewLearningRateStepDecayOptimizer.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	NewLearningRateStepDecayOptimizer.timeStepToDecay = parameterDictionary.timeStepToDecay or defaultTimeStepToDecay
	
	--------------------------------------------------------------------------------
	
	NewLearningRateStepDecayOptimizer:setCalculateFunction(function(learningRate, costFunctionDerivativeTensor)
		
		local currentLearningRate = NewLearningRateStepDecayOptimizer.optimizerInternalParameterArray[1] or learningRate
		
		local currentTimeStep = NewLearningRateStepDecayOptimizer.optimizerInternalParameterArray[2] or 0

		currentTimeStep += 1
		
		if ((currentTimeStep % NewLearningRateStepDecayOptimizer.timeStepToDecay) == 0) then
			
			currentLearningRate *= NewLearningRateStepDecayOptimizer.decayRate
			
		end
		
		costFunctionDerivativeTensor = AqwamTensorLibrary:multiply(currentLearningRate, costFunctionDerivativeTensor)
		
		NewLearningRateStepDecayOptimizer.optimizerInternalParameterArray = {currentLearningRate, currentTimeStep}

		return costFunctionDerivativeTensor
		
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