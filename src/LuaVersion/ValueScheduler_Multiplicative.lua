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

MultiplicativeValueScheduler = {}

MultiplicativeValueScheduler.__index = MultiplicativeValueScheduler

setmetatable(MultiplicativeValueScheduler, BaseValueScheduler)

function MultiplicativeValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMultiplicativeValueScheduler = BaseValueScheduler.new(parameterDictionary)
	
	setmetatable(NewMultiplicativeValueScheduler, MultiplicativeValueScheduler)
	
	NewMultiplicativeValueScheduler:setName("Multiplicative")
	
	local functionToRun = parameterDictionary.functionToRun
	
	if (not functionToRun) then error("No function to run.") end
	
	NewMultiplicativeValueScheduler.functionToRun = functionToRun
	
	--------------------------------------------------------------------------------
	
	NewMultiplicativeValueScheduler:setCalculateFunction(function(value, timeValue)

		return (value * NewMultiplicativeValueScheduler.functionToRun(timeValue))
		
	end)
	
	return NewMultiplicativeValueScheduler
	
end

function MultiplicativeValueScheduler:setFunctionToRun(functionToRun)
	
	self.functionToRun = functionToRun
	
end

return MultiplicativeValueScheduler
