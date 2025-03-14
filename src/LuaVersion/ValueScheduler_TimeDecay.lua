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

TimeDecayValueScheduler = {}

TimeDecayValueScheduler.__index = TimeDecayValueScheduler

setmetatable(TimeDecayValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

function TimeDecayValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTimeDecayValueScheduler = BaseValueScheduler.new()
	
	setmetatable(NewTimeDecayValueScheduler, TimeDecayValueScheduler)
	
	NewTimeDecayValueScheduler:setName("TimeDecay")
	
	NewTimeDecayValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewTimeDecayValueScheduler:setCalculateFunction(function(value)
		
		local currentValue

		local currentTimeStep

		local valueSchedulerInternalParameterArray = NewTimeDecayValueScheduler.valueSchedulerInternalParameterArray

		if (valueSchedulerInternalParameterArray) then

			currentValue = valueSchedulerInternalParameterArray[1]

			currentTimeStep = valueSchedulerInternalParameterArray[2]

		end
		
		currentValue = currentValue or value

		currentTimeStep = currentTimeStep or 0
		
		currentTimeStep = currentTimeStep + 1
			
		currentValue = currentValue * NewTimeDecayValueScheduler.decayRate
		
		NewTimeDecayValueScheduler.valueSchedulerInternalParameterArray = {currentValue, currentTimeStep}

		return currentValue
		
	end)
	
	return NewTimeDecayValueScheduler
	
end

function TimeDecayValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return TimeDecayValueScheduler