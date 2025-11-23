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

local InverseSquareRootValueScheduler = {}

InverseSquareRootValueScheduler.__index = InverseSquareRootValueScheduler

setmetatable(InverseSquareRootValueScheduler, BaseValueScheduler)

function InverseSquareRootValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewInverseSquareRootValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewInverseSquareRootValueScheduler, InverseSquareRootValueScheduler)
	
	NewInverseSquareRootValueScheduler:setName("InverseSquareRoot")
	
	--------------------------------------------------------------------------------
	
	NewInverseSquareRootValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value / math.pow(timeValue, 0.5))
		
	end)
	
	return NewInverseSquareRootValueScheduler
	
end

return InverseSquareRootValueScheduler
