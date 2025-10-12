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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseInstance = require("Core_BaseInstance")

BaseEligibilityTrace = {}

BaseEligibilityTrace.__index = BaseEligibilityTrace

setmetatable(BaseEligibilityTrace, BaseInstance)

local defaultLambda = 0.5

local defaultMode = "StateAction"

function BaseEligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseEligibilityTrace = BaseInstance.new()
	
	setmetatable(NewBaseEligibilityTrace, BaseEligibilityTrace)
	
	NewBaseEligibilityTrace:setName("BaseEligibilityTrace")
	
	NewBaseEligibilityTrace:setClassName("EligibilityTrace")
	
	NewBaseEligibilityTrace.lambda = parameterDictionary.lambda or defaultLambda
	
	NewBaseEligibilityTrace.mode = parameterDictionary.mode or defaultMode
	
	NewBaseEligibilityTrace.eligibilityTraceMatrix = nil
	
	return NewBaseEligibilityTrace
	
end

function BaseEligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray) -- This function is needed because we have double version of reinforcement learning algorithms require separate application of (temporalDifferenceErrorVector * eligibilityTraceMatrix).
	
	local eligibilityTraceMatrix = self.eligibilityTraceMatrix
	
	if (not eligibilityTraceMatrix) then
		
		local mode = self.mode
		
		local selectedDimensionSizeArray
		
		if (mode == "StateAction") then
			
			selectedDimensionSizeArray = dimensionSizeArray
			
		elseif (mode == "State") then
			
			actionIndex = 1
			
			selectedDimensionSizeArray = {dimensionSizeArray[1], 1}
			
		else
			
			error("Unknown mode.")
			
		end
		
		eligibilityTraceMatrix = AqwamTensorLibrary:createTensor(selectedDimensionSizeArray, 0)
		
	end

	eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * self.lambda)
	
	self.eligibilityTraceMatrix = self.incrementFunction(eligibilityTraceMatrix, stateIndex, actionIndex)
	
end

function BaseEligibilityTrace:calculate(temporalDifferenceErrorVector)
	
	return AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, self.eligibilityTraceMatrix)
	
end

function BaseEligibilityTrace:setIncrementFunction(incrementFunction)
	
	self.incrementFunction = incrementFunction
	
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
