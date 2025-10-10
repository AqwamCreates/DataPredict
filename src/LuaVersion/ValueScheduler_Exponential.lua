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

ExponentValueScheduler = {}

ExponentValueScheduler.__index = ExponentValueScheduler

setmetatable(ExponentValueScheduler, BaseValueScheduler)

local defaultDecayRate = 0.5

function ExponentValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewExponentValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewExponentValueScheduler, ExponentValueScheduler)
	
	NewExponentValueScheduler:setName("Exponent")
	
	NewExponentValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewExponentValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value * math.exp(-NewExponentValueScheduler.decayRate * timeValue))
		
	end)
	
	return NewExponentValueScheduler
	
end

function ExponentValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return ExponentValueScheduler
