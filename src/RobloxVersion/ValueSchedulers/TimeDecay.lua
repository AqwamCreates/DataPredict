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

TimeDecayValueScheduler = {}

TimeDecayValueScheduler.__index = TimeDecayValueScheduler

setmetatable(TimeDecayValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

local defaultTimeStepToDecay = 100

function TimeDecayValueScheduler.new(decayRate, timeStepToDecay)
	
	local NewTimeDecayValueScheduler = BaseValueScheduler.new("TimeDecay")
	
	setmetatable(NewTimeDecayValueScheduler, TimeDecayValueScheduler)
	
	NewTimeDecayValueScheduler.decayRate = decayRate or defaultDecayRate
	
	NewTimeDecayValueScheduler.timeStepToDecay = timeStepToDecay or defaultTimeStepToDecay
	
	--------------------------------------------------------------------------------
	
	NewTimeDecayValueScheduler:setCalculateFunction(function(value)
		
		local currentValue

		local currentTimeStep

		local valueSchedulerInternalParameters = NewTimeDecayValueScheduler.valueSchedulerInternalParameters

		if (valueSchedulerInternalParameters) then

			currentValue = valueSchedulerInternalParameters[1]

			currentTimeStep = valueSchedulerInternalParameters[2]

		end
		
		currentValue = currentValue or value

		currentTimeStep = currentTimeStep or 0
		
		currentTimeStep = currentTimeStep + 1
			
		currentValue = currentValue * NewTimeDecayValueScheduler.decayRate
		
		NewTimeDecayValueScheduler.valueSchedulerInternalParameters = {currentValue, currentTimeStep}

		return currentValue
		
	end)
	
	return NewTimeDecayValueScheduler
	
end

function TimeDecayValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return TimeDecayValueScheduler