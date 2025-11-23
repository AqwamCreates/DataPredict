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

local CosineAnnealingValueScheduler = {}

CosineAnnealingValueScheduler.__index = CosineAnnealingValueScheduler

setmetatable(CosineAnnealingValueScheduler, BaseValueScheduler)

local defaultMaximumTimeValue = 5

local defaultMinimumValue = 1

function CosineAnnealingValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewCosineAnnealingValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewCosineAnnealingValueScheduler, CosineAnnealingValueScheduler)
	
	NewCosineAnnealingValueScheduler:setName("CosineAnnealing")
	
	NewCosineAnnealingValueScheduler.maximumTimeValue = parameterDictionary.maximumTimeValue or defaultMaximumTimeValue
	
	NewCosineAnnealingValueScheduler.minimumValue = parameterDictionary.minimumValue or defaultMinimumValue
	
	--------------------------------------------------------------------------------
	
	NewCosineAnnealingValueScheduler:setCalculateFunction(function(value, timeValue)
		
		local minimumValue = NewCosineAnnealingValueScheduler.minimumValue
		
		local multiplyValuePart1 = 1 + math.cos((timeValue * math.pi) / NewCosineAnnealingValueScheduler.maximumTimeValue)
		
		local multiplyValuePart2 = (value - minimumValue)
		
		local multiplyValue = 0.5 * multiplyValuePart1 * multiplyValuePart2
		
		return (minimumValue + multiplyValue)
		
	end)
	
	return NewCosineAnnealingValueScheduler
	
end

function CosineAnnealingValueScheduler:setMaximumTimeValue(maximumTimeValue)
	
	self.maximumTimeValue = maximumTimeValue
	
end

function CosineAnnealingValueScheduler:setMinimumValue(minimumValue)

	self.minimumValue = minimumValue

end

return CosineAnnealingValueScheduler
