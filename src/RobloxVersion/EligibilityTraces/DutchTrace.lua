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

DutchTrace = {}

DutchTrace.__index = DutchTrace

setmetatable(DutchTrace, BaseEligibilityTrace)

local defaultAlpha = 0.5

function DutchTrace.new(parameterDictionary)
	
	local NewDutchTrace = BaseEligibilityTrace.new(parameterDictionary)
	
	setmetatable(NewDutchTrace, DutchTrace)
	
	NewDutchTrace:setName("DutchTrace")
	
	NewDutchTrace.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewDutchTrace:setCalculateFunction(function(eligibilityTraceMatrix, actionIndex)

		eligibilityTraceMatrix[1][actionIndex] = ((1 - NewDutchTrace.alpha) * eligibilityTraceMatrix[1][actionIndex]) + 1
		
		return eligibilityTraceMatrix
		
	end)
	
	return NewDutchTrace
	
end

return DutchTrace
