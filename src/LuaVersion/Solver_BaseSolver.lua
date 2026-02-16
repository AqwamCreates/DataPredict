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

local defaultIsLinear = false

function BaseSolver.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseSolver = BaseInstance.new()
	
	setmetatable(NewBaseSolver, BaseSolver)
	
	NewBaseSolver:setName("BaseSolver")
	
	NewBaseSolver:setClassName("Solver")
	
	NewBaseSolver.isLinear = NewBaseSolver:getValueOrDefaultValue(parameterDictionary.isLinear, defaultIsLinear)
	
	NewBaseSolver.calculateFunction = parameterDictionary.calculateFunction
	
	NewBaseSolver.cache = parameterDictionary.cache
	
	return NewBaseSolver
	
end

function BaseSolver:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseSolver:calculate(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
	
	return self.calculateFunction(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
	
end

function BaseSolver:setCache(cache)
	
	self.cache = cache
	
end

function BaseSolver:getCache(cache)

	return self.cache

end

function BaseSolver:reset()

	self.cache = nil

end

return BaseSolver
