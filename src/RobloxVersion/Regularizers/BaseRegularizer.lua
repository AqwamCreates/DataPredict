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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

BaseRegularizer = {}

BaseRegularizer.__index = BaseRegularizer

setmetatable(BaseRegularizer, BaseInstance)

local defaultLambda = 0.01

local defaultHasBias = "Automatic"

function BaseRegularizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseRegularizer = BaseInstance.new()
	
	setmetatable(NewBaseRegularizer, BaseRegularizer)
	
	NewBaseRegularizer:setName("BaseRegularizer")
	
	NewBaseRegularizer:getClassName("Regularizer")
	
	NewBaseRegularizer.lambda = parameterDictionary.lambda or defaultLambda
	
	NewBaseRegularizer.hasBias = NewBaseRegularizer:getValueOrDefaultValue(parameterDictionary.hasBias, defaultHasBias)
	
	return NewBaseRegularizer
	
end

function BaseRegularizer:adjustModelParameters(ModelParameters)
	
	local hasBias = self.hasBias
	
	if (not hasBias) then return ModelParameters end
	
	local firstRowModelParameters = ModelParameters[1]
	
	if (hasBias == "Automatic") then
		
		for i, value in ipairs(firstRowModelParameters) do
			
			if (value ~= 1) then return ModelParameters end
			
		end
		
	end
	
	local NewModelParameters = self:deepCopyTable(ModelParameters)
	
	firstRowModelParameters = NewModelParameters[1]
	
	for i, _ in ipairs(firstRowModelParameters) do
		
		firstRowModelParameters[i] = 0
		
	end

	return NewModelParameters

end

function BaseRegularizer:calculateCost(ModelParameters)
	
	local CalculateCostFunction = self.CalculateCostFunction
	
	if (CalculateCostFunction) then return CalculateCostFunction(ModelParameters) end
	
end

function BaseRegularizer:setCalculateCostFunction(CalculateCostFunction)

	self.CalculateCostFunction = CalculateCostFunction

end

function BaseRegularizer:calculate(ModelParameters)

	local CalculateFunction = self.CalculateFunction

	if (CalculateFunction) then return CalculateFunction(ModelParameters) end

end

function BaseRegularizer:setCalculateFunction(CalculateFunction)
	
	self.CalculateFunction = CalculateFunction
	
end

function BaseRegularizer:getLambda()
	
	return self.lambda
	
end

function BaseRegularizer:setLambda(lambda)

	self.lambda = lambda

end

return BaseRegularizer
