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

local BaseInstance = require("Core_BaseInstance")

local BaseSolver = {}

BaseSolver.__index = BaseSolver

setmetatable(BaseSolver, BaseInstance)

local defaultIsNonLinear = true

function BaseSolver.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseSolver = BaseInstance.new()
	
	setmetatable(NewBaseSolver, BaseSolver)
	
	NewBaseSolver:setName("BaseSolver")
	
	NewBaseSolver:setClassName("Solver")
	
	NewBaseSolver.isNonLinear = NewBaseSolver:getValueOrDefaultValue(parameterDictionary.isNonLinear, defaultIsNonLinear)
	
	NewBaseSolver.calculateFunction = parameterDictionary.calculateFunction
	
	NewBaseSolver.resetFunction = parameterDictionary.resetFunction
	
	NewBaseSolver.cache = parameterDictionary.cache
	
	return NewBaseSolver
	
end

function BaseSolver:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseSolver:setResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function BaseSolver:setCache(cache)
	
	self.cache = cache
	
end

function BaseSolver:getCache(cache)

	return self.cache

end

function BaseSolver:calculate(weightMatrix, inputMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)

	return self.calculateFunction(weightMatrix, inputMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)

end

function BaseSolver:reset()

	local resetFunction = self.resetFunction

	if (resetFunction) then return resetFunction() end

end

return BaseSolver
