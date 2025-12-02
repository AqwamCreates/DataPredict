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

local BaseEligibilityTrace = {}

BaseEligibilityTrace.__index = BaseEligibilityTrace

setmetatable(BaseEligibilityTrace, BaseInstance)

local defaultLambda = 0.5

local defaultMode = "StateAction"

local defaultIsStateFromAction = false

local defaultIsTabular = false

local safeguardedInverseFunction = function(denominator)
	
	if (denominator == 0) then return 0 end
	
	return (1 / denominator)
	
end

local function invertMatrix(matrix)
	
	return AqwamTensorLibrary:applyFunction(safeguardedInverseFunction, matrix)
	
end

function BaseEligibilityTrace.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseEligibilityTrace = BaseInstance.new()
	
	setmetatable(NewBaseEligibilityTrace, BaseEligibilityTrace)
	
	NewBaseEligibilityTrace:setName("BaseEligibilityTrace")
	
	NewBaseEligibilityTrace:setClassName("EligibilityTrace")
	
	NewBaseEligibilityTrace.lambda = parameterDictionary.lambda or defaultLambda
	
	NewBaseEligibilityTrace.mode = parameterDictionary.mode or defaultMode
	
	NewBaseEligibilityTrace.isStateFromAction = NewBaseEligibilityTrace:getValueOrDefaultValue(parameterDictionary.isStateFromAction, defaultIsStateFromAction)
	
	NewBaseEligibilityTrace.isTabular  = NewBaseEligibilityTrace:getValueOrDefaultValue(parameterDictionary.isTabular, defaultIsTabular)
	
	NewBaseEligibilityTrace.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix
	
	NewBaseEligibilityTrace.stateActionEligibilityTraceMatrix = parameterDictionary.stateActionEligibilityTraceMatrix
	
	return NewBaseEligibilityTrace
	
end

function BaseEligibilityTrace:increment(stateIndex, actionIndex, discountFactor, dimensionSizeArray) -- This function is needed because we have double version of reinforcement learning algorithms require separate application of (temporalDifferenceErrorVector * eligibilityTraceMatrix).
	
	local lambda = self.lambda
	
	local mode = self.mode
	
	local isStateFromAction = self.isStateFromAction
	
	local eligibilityTraceMatrix = self.eligibilityTraceMatrix
	
	local stateActionEligibilityTraceMatrix = self.stateActionEligibilityTraceMatrix
	
	if (mode == "State") then 
		
		actionIndex = 1
		
	elseif (mode == "Action") then
		
		stateIndex = 1
		
	end
	
	if (not eligibilityTraceMatrix) then
		
		local selectedDimensionSizeArray
		
		if (mode == "StateAction") then
			
			selectedDimensionSizeArray = dimensionSizeArray
			
		elseif (mode == "State") then
			
			selectedDimensionSizeArray = {dimensionSizeArray[1], 1}
			
		elseif (mode == "Action") then
			
			selectedDimensionSizeArray = {1, dimensionSizeArray[2]}
			
		else
			
			error("Unknown mode.")
			
		end
		
		eligibilityTraceMatrix = AqwamTensorLibrary:createTensor(selectedDimensionSizeArray, 0)
		
	end
	
	if (isStateFromAction) and (mode == "Action") then
		
		local stateEligibilityTraceMatrix 
		
		if (self.isTabular) then
			
			if (not stateActionEligibilityTraceMatrix) then stateActionEligibilityTraceMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, 0) end

			stateActionEligibilityTraceMatrix = self.incrementFunction(stateActionEligibilityTraceMatrix, stateIndex, actionIndex)

			stateEligibilityTraceMatrix = AqwamTensorLibrary:divide(stateActionEligibilityTraceMatrix, eligibilityTraceMatrix)

			self.stateActionEligibilityTraceMatrix = stateActionEligibilityTraceMatrix
			
		else
			
			-- Currently, we assume that the algorithm will only visit the state / state-action once due to continuous values and the high dimensionality of the states.
			
			local actionEligibilityTraceMatrix = invertMatrix(eligibilityTraceMatrix) -- Assume that the input is from E(s) = E(a)^-1, hence E(a) = E(s)^-1.

			actionEligibilityTraceMatrix = self.incrementFunction(actionEligibilityTraceMatrix, stateIndex, actionIndex) 

			stateEligibilityTraceMatrix = invertMatrix(actionEligibilityTraceMatrix)
			
		end
		
		stateEligibilityTraceMatrix = AqwamTensorLibrary:multiply(stateEligibilityTraceMatrix, discountFactor * lambda)
		
		self.eligibilityTraceMatrix = stateEligibilityTraceMatrix
		
	elseif (isStateFromAction) and (mode ~= "Action") then
		
		error("If isStateFromAction is set to true, it can only accept \"action\" mode.")
		
	elseif (mode ~= "State") and (mode ~= "Action") and (mode ~= "StateAction") then
		
		error("Invalid mode.")
		
	else
		
		eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)
		
		self.eligibilityTraceMatrix = self.incrementFunction(eligibilityTraceMatrix, stateIndex, actionIndex)
		
	end

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
	
	self.stateActionEligibilityTraceMatrix = nil
	
end

return BaseEligibilityTrace
