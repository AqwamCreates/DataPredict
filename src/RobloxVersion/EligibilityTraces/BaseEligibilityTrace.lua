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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

BaseEligibilityTrace = {}

BaseEligibilityTrace.__index = BaseEligibilityTrace

setmetatable(BaseEligibilityTrace, BaseInstance)

local defaultLambda = 0.5

function BaseEligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseEligibilityTrace = BaseInstance.new()
	
	setmetatable(NewBaseEligibilityTrace, BaseEligibilityTrace)
	
	NewBaseEligibilityTrace:setName("BaseEligibilityTrace")
	
	NewBaseEligibilityTrace:getClassName("EligibilityTrace")
	
	NewBaseEligibilityTrace.lambda = parameterDictionary.lambda or defaultLambda
	
	NewBaseEligibilityTrace.eligibilityTraceMatrix = nil
	
	return NewBaseEligibilityTrace
	
end

function BaseEligibilityTrace:calculate(temporalDifferenceErrorVector, actionIndex, discountFactor)
	
	if (self.CalculateFunction) then return self.CalculateFunction(temporalDifferenceErrorVector, actionIndex, discountFactor) end
	
end

function BaseEligibilityTrace:setCalculateFunction(CalculateFunction)
	
	self.CalculateFunction = CalculateFunction
	
end

function BaseEligibilityTrace:getLambda()
	
	return self.lambda
	
end

function BaseEligibilityTrace:setLambda(lambda)

	self.lambda = lambda

end

function BaseEligibilityTrace:reset()
	
	self.eligibilityTraceMatrix = nil
	
end

return BaseEligibilityTrace
