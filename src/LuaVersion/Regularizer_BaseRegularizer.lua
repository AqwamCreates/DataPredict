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

local BaseRegularizer = {}

BaseRegularizer.__index = BaseRegularizer

setmetatable(BaseRegularizer, BaseInstance)

local defaultLambda = 0.01

local defaultHasBias = "Automatic"

function BaseRegularizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseRegularizer = BaseInstance.new()
	
	setmetatable(NewBaseRegularizer, BaseRegularizer)
	
	NewBaseRegularizer:setName("BaseRegularizer")
	
	NewBaseRegularizer:setClassName("Regularizer")
	
	NewBaseRegularizer.lambda = parameterDictionary.lambda or defaultLambda
	
	NewBaseRegularizer.hasBias = NewBaseRegularizer:getValueOrDefaultValue(parameterDictionary.hasBias, defaultHasBias)
	
	return NewBaseRegularizer
	
end

function BaseRegularizer:adjustWeightMatrix(weightMatrix)
	
	local hasBias = self.hasBias
	
	if (not hasBias) then return weightMatrix end
	
	local firstRowWeightMatrix = weightMatrix[1]
	
	if (hasBias == "Automatic") then
		
		for i, value in ipairs(firstRowWeightMatrix) do
			
			if (value ~= 1) then return weightMatrix end
			
		end
		
	end
	
	local newWeightMatrix = self:deepCopyTable(weightMatrix)
	
	firstRowWeightMatrix = newWeightMatrix[1]
	
	for i, _ in ipairs(firstRowWeightMatrix) do
		
		firstRowWeightMatrix[i] = 0
		
	end

	return newWeightMatrix

end

function BaseRegularizer:calculateCost(weightMatrix)
	
	return self.calculateCostFunction(weightMatrix)
	
end

function BaseRegularizer:setCalculateCostFunction(calculateCostFunction)

	self.calculateCostFunction = calculateCostFunction

end

function BaseRegularizer:calculate(weightMatrix)

	return self.calculateFunction(weightMatrix)

end

function BaseRegularizer:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseRegularizer:getLambda()
	
	return self.lambda
	
end

function BaseRegularizer:setLambda(lambda)

	self.lambda = lambda

end

return BaseRegularizer
