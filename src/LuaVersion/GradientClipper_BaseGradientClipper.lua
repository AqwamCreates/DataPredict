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

BaseGradientClipper = {}

BaseGradientClipper.__index = BaseGradientClipper

setmetatable(BaseGradientClipper, BaseInstance)

local defaultValue = 0

function BaseGradientClipper.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseGradientClipper = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewBaseGradientClipper, BaseGradientClipper)
	
	NewBaseGradientClipper:setName("BaseGradientClipper")

	NewBaseGradientClipper:setClassName("GradientClipper")
	
	NewBaseGradientClipper.Optimizer = parameterDictionary.Optimizer
	
	NewBaseGradientClipper.clipFunction = parameterDictionary.clipFunction
	
	return NewBaseGradientClipper
	
end

function BaseGradientClipper:calculate(learningRate, costFunctionDerivativeMatrix, ...)
	
	local Optimizer = self.Optimizer
	
	local costFunctionDerivativeMatrix = self.clipFunction(costFunctionDerivativeMatrix)
	
	if (Optimizer) then
		
		costFunctionDerivativeMatrix = Optimizer:calculate(learningRate, costFunctionDerivativeMatrix, ...)
		
	else
		
		costFunctionDerivativeMatrix =  AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrix)
		
	end
	
	return costFunctionDerivativeMatrix
	
end

function BaseGradientClipper:setClipFunction(clipFunction)
	
	self.clipFunction = clipFunction
	
end

function BaseGradientClipper:reset()
	
	local Optimizer = self.Optimizer

	if (Optimizer) then Optimizer:reset() end

end

return BaseGradientClipper
