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

local BaseInstance = require("Cores_BaseInstance")

BaseRegularizer = {}

BaseRegularizer.__index = BaseRegularizer

setmetatable(BaseRegularizer, BaseInstance)

local defaultLambda = 0.01

function BaseRegularizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseRegularizer = BaseInstance.new()
	
	setmetatable(NewBaseRegularizer, BaseRegularizer)
	
	NewBaseRegularizer:setName("BaseRegularizer")
	
	NewBaseRegularizer:getClassName("Regularizer")
	
	NewBaseRegularizer.lambda = parameterDictionary.lambda or defaultLambda
	
	return NewBaseRegularizer
	
end

function BaseRegularizer:calculate(weightTensor)
	
	if (self.CalculateFunction) then return self.CalculateFunction(weightTensor) end
	
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
