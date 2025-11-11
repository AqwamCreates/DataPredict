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

ClipValueGradientClipper = {}

ClipValueGradientClipper.__index = ClipValueGradientClipper

setmetatable(ClipValueGradientClipper, BaseGradientClipper)

local defaultMinimumValue = -1

local defaultMaximumValue = 1

function ClipValueGradientClipper.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewClipValueGradientClipper = BaseGradientClipper.new(parameterDictionary)
	
	setmetatable(NewClipValueGradientClipper, ClipValueGradientClipper)
	
	NewClipValueGradientClipper:setName("ClipValue")
	
	NewClipValueGradientClipper.minimumValue = parameterDictionary.minimumValue or defaultMinimumValue
	
	NewClipValueGradientClipper.maximumValue = parameterDictionary.maximumValue or defaultMaximumValue
	
	--------------------------------------------------------------------------------
	
	NewClipValueGradientClipper:setClipFunction(function(costFunctionDerivativeMatrix)
		
		local functionToApply = function(value) math.clamp(value, NewClipValueGradientClipper.minimumValue, NewClipValueGradientClipper.maximumValue) end
		
		return AqwamTensorLibrary:applyFunction(functionToApply, costFunctionDerivativeMatrix)
		
	end)
	
	return NewClipValueGradientClipper
	
end

function ClipValueGradientClipper:setMinimumValue(minimumValue)

	self.minimumValue = minimumValue

end

function ClipValueGradientClipper:setMaximumValue(maximumValue)
	
	self.maximumValue = maximumValue
	
end

return ClipValueGradientClipper
