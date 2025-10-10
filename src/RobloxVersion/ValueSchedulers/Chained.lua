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

local BaseValueScheduler = require(script.Parent.BaseValueScheduler)

ChainedValueScheduler = {}

ChainedValueScheduler.__index = ChainedValueScheduler

setmetatable(ChainedValueScheduler, BaseValueScheduler)

function ChainedValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewChainedValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewChainedValueScheduler, ChainedValueScheduler)
	
	NewChainedValueScheduler:setName("Chained")
	
	local ValueSchedulerArray = parameterDictionary.ValueSchedulerArray
	
	if (not ValueSchedulerArray) then error("No value scheduler array.") end
	
	if (#ValueSchedulerArray <= 0) then error("No value scheduler.") end
	
	NewChainedValueScheduler.ValueSchedulerArray = ValueSchedulerArray
	
	--------------------------------------------------------------------------------
	
	NewChainedValueScheduler:setCalculateFunction(function(value, timeValue)
		
		for _, ValueScheduler in ipairs(NewChainedValueScheduler.ValueSchedulerArray) do
			
			value = ValueScheduler:calculate(value, timeValue)
			
		end

		return value
		
	end)
	
	return NewChainedValueScheduler
	
end

function ChainedValueScheduler:setValueSchedulerArray(ValueSchedulerArray)
	
	self.ValueSchedulerArray = ValueSchedulerArray
	
end

return ChainedValueScheduler
