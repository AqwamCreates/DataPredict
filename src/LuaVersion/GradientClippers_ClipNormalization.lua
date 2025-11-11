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

local BaseGradientClipper = require("GradientClipper_BaseGradientClipper")

ClipNormalizationGradientClipper = {}

ClipNormalizationGradientClipper.__index = ClipNormalizationGradientClipper

setmetatable(ClipNormalizationGradientClipper, BaseGradientClipper)

local defaultNormalizationValue = 2

function ClipNormalizationGradientClipper.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewClipNormalizationGradientClipper = BaseGradientClipper.new(parameterDictionary)
	
	setmetatable(NewClipNormalizationGradientClipper, ClipNormalizationGradientClipper)
	
	NewClipNormalizationGradientClipper:setName("ClipNormalization")
	
	local normalizationValue = parameterDictionary.normalizationValue or defaultNormalizationValue
	
	NewClipNormalizationGradientClipper.normalizationValue = normalizationValue
	
	NewClipNormalizationGradientClipper.maximumNormalizationValue = parameterDictionary.maximumNormalizationValue or normalizationValue
	
	--------------------------------------------------------------------------------
	
	NewClipNormalizationGradientClipper:setClipFunction(function(costFunctionDerivativeMatrix)
		
		local normalizationValue = NewClipNormalizationGradientClipper.normalizationValue
		
		local maximumNormalizationValue = NewClipNormalizationGradientClipper.maximumNormalizationValue
		
		local squaredCostFunctionDerivativeMatrix = AqwamTensorLibrary:power(costFunctionDerivativeMatrix, normalizationValue)
		
		local sumSquaredCostFunctionDerivativeMatrix = AqwamTensorLibrary:sum(squaredCostFunctionDerivativeMatrix)
		
		local currentNormalizationValue = math.pow(sumSquaredCostFunctionDerivativeMatrix, (1 / normalizationValue))
		
		if (currentNormalizationValue ~= 0) then
			
			costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(costFunctionDerivativeMatrix, (maximumNormalizationValue / currentNormalizationValue))
		
		end
		
		return costFunctionDerivativeMatrix
		
	end)
	
	return NewClipNormalizationGradientClipper
	
end

function ClipNormalizationGradientClipper:setNormalizationValue(normalizationValue)

	self.normalizationValue = normalizationValue

end

function ClipNormalizationGradientClipper:setMaximumNormalizationValue(maximumNormalizationValue)

	self.maximumNormalizationValue = maximumNormalizationValue

end

return ClipNormalizationGradientClipper
