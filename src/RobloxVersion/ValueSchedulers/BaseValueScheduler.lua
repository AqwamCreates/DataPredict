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

BaseValueScheduler = {}

BaseValueScheduler.__index = BaseValueScheduler

setmetatable(BaseValueScheduler, BaseInstance)

local defaultValue = 0

function BaseValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseValueScheduler = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewBaseValueScheduler, BaseValueScheduler)
	
	NewBaseValueScheduler:setName("BaseValueScheduler")

	NewBaseValueScheduler:setClassName("ValueScheduler")
	
	NewBaseValueScheduler.calculateFunction = parameterDictionary.calculateFunction
	
	NewBaseValueScheduler.timeValue = parameterDictionary.timeValue or defaultValue
	
	return NewBaseValueScheduler
	
end

function BaseValueScheduler:calculate(valueToSchedule, valueToScale)
	
	local timeValue = self.timeValue
	
	timeValue = timeValue + 1
	
	self.timeValue = timeValue
	
	valueToSchedule = self.calculateFunction(valueToSchedule, timeValue)
	
	if (not valueToScale) then return valueToSchedule end
	
	return AqwamTensorLibrary:multiply(valueToSchedule, valueToScale)
	
end

function BaseValueScheduler:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseValueScheduler:reset()

	self.timeValue = 0

end

return BaseValueScheduler
