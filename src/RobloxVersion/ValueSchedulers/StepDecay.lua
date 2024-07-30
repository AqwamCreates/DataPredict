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

local BaseValueScheduler = require(script.Parent.BaseValueScheduler)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

StepDecayValueScheduler = {}

StepDecayValueScheduler.__index = StepDecayValueScheduler

setmetatable(StepDecayValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

local defaultTimeStepToDecay = 100

function StepDecayValueScheduler.new(decayRate, timeStepToDecay)
	
	local NewStepDecayValueScheduler = BaseValueScheduler.new("StepDecay")
	
	setmetatable(NewStepDecayValueScheduler, StepDecayValueScheduler)
	
	NewStepDecayValueScheduler.decayRate = decayRate or defaultDecayRate
	
	NewStepDecayValueScheduler.timeStepToDecay = timeStepToDecay or defaultTimeStepToDecay
	
	--------------------------------------------------------------------------------
	
	NewStepDecayValueScheduler:setCalculateFunction(function(value)
		
		local currentValue
		
		local currentTimeStep
		
		local valueSchedulerInternalParameters = NewStepDecayValueScheduler.valueSchedulerInternalParameters
		
		if (valueSchedulerInternalParameters) then
			
			currentValue = valueSchedulerInternalParameters[1]
			
			currentTimeStep = valueSchedulerInternalParameters[2]
			
		end
		
		currentValue = currentValue or value
		
		currentTimeStep = currentTimeStep or 1
		
		if ((currentTimeStep % NewStepDecayValueScheduler.timeStepToDecay) == 0) then currentValue *= NewStepDecayValueScheduler.decayRate end
		
		currentTimeStep = currentTimeStep + 1
		
		NewStepDecayValueScheduler.valueSchedulerInternalParameters = {currentValue, currentTimeStep}

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
