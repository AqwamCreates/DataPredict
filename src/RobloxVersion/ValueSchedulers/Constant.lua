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

ConstantValueScheduler = {}

ConstantValueScheduler.__index = ConstantValueScheduler

setmetatable(ConstantValueScheduler, BaseValueScheduler)

local defaultTimeValueToDecay = 1

local defaultDecayRate = 0.5

function ConstantValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewConstantValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewConstantValueScheduler, ConstantValueScheduler)
	
	NewConstantValueScheduler:setName("Constant")
	
	NewConstantValueScheduler.timeValueToDecay = parameterDictionary.timeValueToDecay or defaultTimeValueToDecay
	
	NewConstantValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewConstantValueScheduler:setCalculateFunction(function(value, timeValue)
		
		if (timeValue <= NewConstantValueScheduler.timeValueToDecay) then return value end

		return (value * NewConstantValueScheduler.decayRate)
		
	end)
	
	return NewConstantValueScheduler
	
end

function ConstantValueScheduler:setTimeValueToDecay(timeValueToDecay)

	self.timeValueToDecay = timeValueToDecay

end

function ConstantValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return ConstantValueScheduler
