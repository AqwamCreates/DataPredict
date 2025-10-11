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

InverseTimeDecayValueScheduler = {}

InverseTimeDecayValueScheduler.__index = InverseTimeDecayValueScheduler

setmetatable(InverseTimeDecayValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

function InverseTimeDecayValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewInverseTimeDecayValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewInverseTimeDecayValueScheduler, InverseTimeDecayValueScheduler)
	
	NewInverseTimeDecayValueScheduler:setName("InverseTimeDecay")
	
	NewInverseTimeDecayValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewInverseTimeDecayValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value / (1 + (NewInverseTimeDecayValueScheduler.decayRate * timeValue)))
		
	end)
	
	return NewInverseTimeDecayValueScheduler
	
end

function InverseTimeDecayValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return InverseTimeDecayValueScheduler
