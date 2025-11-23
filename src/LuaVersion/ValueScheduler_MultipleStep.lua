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

local MultipleStepValueScheduler = {}

MultipleStepValueScheduler.__index = MultipleStepValueScheduler

setmetatable(MultipleStepValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

function MultipleStepValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMultipleStepValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewMultipleStepValueScheduler, MultipleStepValueScheduler)
	
	NewMultipleStepValueScheduler:setName("MultipleStep")
	
	local timeValueArray = parameterDictionary.timeValueArray
	
	if (not timeValueArray) then error("No time value array.") end
	
	if (#timeValueArray <= 0) then error("No time value.") end
	
	NewMultipleStepValueScheduler.timeValueArray = timeValueArray
	
	NewMultipleStepValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewMultipleStepValueScheduler:setCalculateFunction(function(value, timeValue)
		
		local decayCount = 0
		
		for i, timeValueMilestone in ipairs(NewMultipleStepValueScheduler.timeValueArray) do
			
			if (timeValue <= timeValueMilestone) then break end
				
			decayCount = decayCount + 1
			
		end

		return (value * math.pow(NewMultipleStepValueScheduler.decayRate, decayCount))
		
	end)
	
	return NewMultipleStepValueScheduler
	
end

function MultipleStepValueScheduler:setTimeValueArray(timeValueArray)

	self.timeValueArray = timeValueArray

end

function MultipleStepValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return MultipleStepValueScheduler
