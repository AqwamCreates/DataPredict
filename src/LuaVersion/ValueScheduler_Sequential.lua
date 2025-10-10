--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

SequentialValueScheduler = {}

SequentialValueScheduler.__index = SequentialValueScheduler

setmetatable(SequentialValueScheduler, BaseValueScheduler)

function SequentialValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewSequentialValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewSequentialValueScheduler, SequentialValueScheduler)
	
	NewSequentialValueScheduler:setName("Sequential")
	
	local ValueSchedulerArray = parameterDictionary.ValueSchedulerArray
	
	local timeValueArray = parameterDictionary.timeValueArray
	
	if (not ValueSchedulerArray) then error("No value scheduler array.") end
	
	local numberOfValueSchedulers = #ValueSchedulerArray
	
	if (numberOfValueSchedulers <= 0) then error("No value scheduler.") end
	
	if (not timeValueArray) then error("No time value array.") end
	
	local numberOfTimeValueArray = #timeValueArray

	if (numberOfTimeValueArray <= 0) then error("No time value.") end
	
	if (numberOfValueSchedulers ~= numberOfTimeValueArray) then error("The number of value schedulers is not equal to the number of time values.") end
	
	NewSequentialValueScheduler.ValueSchedulerArray = ValueSchedulerArray
	
	NewSequentialValueScheduler.timeValueArray = timeValueArray
	
	--------------------------------------------------------------------------------
	
	NewSequentialValueScheduler:setCalculateFunction(function(value, timeValue)
		
		local ValueSchedulerArray = NewSequentialValueScheduler.ValueSchedulerArray
		
		local timeValueArray = NewSequentialValueScheduler.timeValueArray
		
		for i, ValueScheduler in ipairs(ValueSchedulerArray) do
			
			if (timeValue <= timeValueArray[i]) then

				return ValueScheduler:calculate(value, timeValue)

			elseif (i == #ValueSchedulerArray) then

				return ValueScheduler:calculate(value, timeValue) 

			end
			
		end
		
	end)
	
	return NewSequentialValueScheduler
	
end

function SequentialValueScheduler:setValueSchedulerArray(ValueSchedulerArray)
	
	self.ValueSchedulerArray = ValueSchedulerArray
	
end

function SequentialValueScheduler:setTimeValueArray(timeValueArray)

	self.timeValueArray = timeValueArray

end

return SequentialValueScheduler
