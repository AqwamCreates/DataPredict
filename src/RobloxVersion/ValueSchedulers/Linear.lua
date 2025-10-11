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

LinearValueScheduler = {}

LinearValueScheduler.__index = LinearValueScheduler

setmetatable(LinearValueScheduler, BaseValueScheduler)

local defaultTimeValue = 5

local defaultStartFactor = 1/3

local defaultEndFactor = 1

function LinearValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewLinearValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewLinearValueScheduler, LinearValueScheduler)
	
	NewLinearValueScheduler:setName("Linear")
	
	NewLinearValueScheduler.timeValue = parameterDictionary.timeValue or defaultTimeValue
	
	NewLinearValueScheduler.startFactor = parameterDictionary.startFactor or defaultStartFactor
	
	NewLinearValueScheduler.endFactor = parameterDictionary.endFactor or defaultEndFactor
	
	--------------------------------------------------------------------------------
	
	NewLinearValueScheduler:setCalculateFunction(function(value, timeValue)
		
		local otherTimeValue = NewLinearValueScheduler.timeValue
		
		local endFactor = NewLinearValueScheduler.endFactor
		
		if (timeValue >= otherTimeValue) then return (value * endFactor) end
		
		local startFactor = NewLinearValueScheduler.startFactor
		
		local factor = startFactor + ((endFactor - startFactor) * (timeValue / otherTimeValue))

		return (value * factor)
		
	end)
	
	return NewLinearValueScheduler
	
end

function LinearValueScheduler:setTimeValue(timeValue)

	self.timeValue = timeValue

end

function LinearValueScheduler:setStartFactor(startFactor)
	
	self.startFactor = startFactor
	
end

function LinearValueScheduler:setEndFactor(endFactor)
	
	self.endFactor = endFactor
	
end

return LinearValueScheduler
