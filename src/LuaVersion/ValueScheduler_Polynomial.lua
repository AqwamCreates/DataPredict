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

PolynomialValueScheduler = {}

PolynomialValueScheduler.__index = PolynomialValueScheduler

setmetatable(PolynomialValueScheduler, BaseValueScheduler)

local defaultTotalTimeValue = 5

local defaultPower = 1

function PolynomialValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewPolynomialValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewPolynomialValueScheduler, PolynomialValueScheduler)
	
	NewPolynomialValueScheduler:setName("Polynomial")
	
	NewPolynomialValueScheduler.totalTimeValue = parameterDictionary.totalTimeValue or defaultTotalTimeValue
	
	NewPolynomialValueScheduler.power = parameterDictionary.power or defaultPower
	
	--------------------------------------------------------------------------------
	
	NewPolynomialValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value * math.pow((1 - (timeValue / NewPolynomialValueScheduler.totalTimeValue)), NewPolynomialValueScheduler.defaultPower))
		
	end)
	
	return NewPolynomialValueScheduler
	
end

function PolynomialValueScheduler:setTotalTimeValue(totalTimeValue)
	
	self.totalTimeValue = totalTimeValue
	
end

function PolynomialValueScheduler:setPower(power)
	
	self.power = power
	
end

return PolynomialValueScheduler
