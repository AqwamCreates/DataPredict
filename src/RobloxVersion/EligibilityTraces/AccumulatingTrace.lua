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

local BaseEligibilityTrace = require(script.Parent.BaseEligibilityTrace)

AccumulatingTrace = {}

AccumulatingTrace.__index = AccumulatingTrace

setmetatable(AccumulatingTrace, BaseEligibilityTrace)

function AccumulatingTrace.new(parameterDictionary)
	
	local NewAccumulatingTrace = BaseEligibilityTrace.new(parameterDictionary)
	
	setmetatable(NewAccumulatingTrace, AccumulatingTrace)
	
	NewAccumulatingTrace:setName("AccumulatingTrace")
	
	NewAccumulatingTrace:setIncrementFunction(function(eligibilityTraceMatrix, actionIndex)

		eligibilityTraceMatrix[1][actionIndex] = eligibilityTraceMatrix[1][actionIndex] + 1
		
		return eligibilityTraceMatrix
		
	end)
	
	return NewAccumulatingTrace
	
end

return AccumulatingTrace
