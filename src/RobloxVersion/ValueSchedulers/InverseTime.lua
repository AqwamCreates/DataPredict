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

local InverseTimeValueScheduler = {}

InverseTimeValueScheduler.__index = InverseTimeValueScheduler

setmetatable(InverseTimeValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

function InverseTimeValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewInverseTimeValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewInverseTimeValueScheduler, InverseTimeValueScheduler)
	
	NewInverseTimeValueScheduler:setName("InverseTime")
	
	NewInverseTimeValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewInverseTimeValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value / (1 + (NewInverseTimeValueScheduler.decayRate * timeValue)))
		
	end)
	
	return NewInverseTimeValueScheduler
	
end

function InverseTimeValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return InverseTimeValueScheduler
