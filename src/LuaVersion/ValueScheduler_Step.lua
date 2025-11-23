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

local StepValueScheduler = {}

StepValueScheduler.__index = StepValueScheduler

setmetatable(StepValueScheduler, BaseValueScheduler)

local defaultTimeValue = 100

local defaultDecayRate = 0.5

function StepValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewStepValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewStepValueScheduler, StepValueScheduler)
	
	NewStepValueScheduler:setName("Step")
	
	NewStepValueScheduler.timeValue = parameterDictionary.timeValue or defaultTimeValue
	
	NewStepValueScheduler.decayRate = parameterDictionary.decayRate or defaultDecayRate
	
	--------------------------------------------------------------------------------
	
	NewStepValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value * math.pow(NewStepValueScheduler.decayRate, (math.floor(timeValue / NewStepValueScheduler.timeValue))))
		
	end)
	
	return NewStepValueScheduler
	
end

function StepValueScheduler:setTimeValue(timeValue)

	self.timeValue = timeValue

end

function StepValueScheduler:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return StepValueScheduler
