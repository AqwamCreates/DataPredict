--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseValueScheduler = require("ValueScheduler_BaseValueScheduler")

StepDecayValueScheduler = {}

StepDecayValueScheduler.__index = StepDecayValueScheduler

setmetatable(StepDecayValueScheduler, BaseValueScheduler)

local defaultTimeStepToDecay = 100

local defaultDecayRate = 0.5

function StepDecayValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewStepDecayValueScheduler = BaseValueScheduler.new()
	
	setmetatable(NewStepDecayValueScheduler, StepDecayValueScheduler)
	
	NewStepDecayValueScheduler:setName("StepDecay")

	NewStepDecayValueScheduler.timeStepToDecay = parameterDictionary.timeStepToDecay or defaultTimeStepToDecay
	
	NewStepDecayValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewStepDecayValueScheduler:setCalculateFunction(function(value)
		
		local currentValue
		
		local currentTimeStep
		
		local valueSchedulerInternalParameterArray = NewStepDecayValueScheduler.valueSchedulerInternalParameterArray
		
		if (valueSchedulerInternalParameterArray) then
			
			currentValue = valueSchedulerInternalParameterArray[1]
			
			currentTimeStep = valueSchedulerInternalParameterArray[2]
			
		end
		
		currentValue = currentValue or value
		
		currentTimeStep = currentTimeStep or 0
		
		currentTimeStep = currentTimeStep + 1
		
		if ((currentTimeStep % NewStepDecayValueScheduler.timeStepToDecay) == 0) then currentValue = currentValue * NewStepDecayValueScheduler.decayRate end
		
		NewStepDecayValueScheduler.valueSchedulerInternalParameterArray = {currentValue, currentTimeStep}

		return currentValue
		
	end)
	
	return NewStepDecayValueScheduler
	
end

function StepDecayValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

function StepDecayValueScheduler:setTimeStepToDecay(timeStepToDecay)
	
	self.timeStepToDecay = timeStepToDecay
	
end

return StepDecayValueScheduler