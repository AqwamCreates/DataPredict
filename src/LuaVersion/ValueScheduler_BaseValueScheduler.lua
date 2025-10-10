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

BaseValueScheduler = {}

BaseValueScheduler.__index = BaseValueScheduler

setmetatable(BaseValueScheduler, BaseInstance)

function BaseValueScheduler.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseValueScheduler = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewBaseValueScheduler, BaseValueScheduler)
	
	NewBaseValueScheduler:setName("BaseValueScheduler")

	NewBaseValueScheduler:setClassName("ValueScheduler")
	
	NewBaseValueScheduler.calculateFunction = parameterDictionary.calculateFunction
	
	NewBaseValueScheduler.valueSchedulerInternalParameterArray = parameterDictionary.valueSchedulerInternalParameterArray
	
	return NewBaseValueScheduler
	
end

function BaseValueScheduler:calculate(valueToSchedule, valueToScale)
	
	valueToSchedule = self.calculateFunction(valueToSchedule) 
	
	if (not valueToScale) then return valueToSchedule end
	
	return AqwamTensorLibrary:multiply(valueToSchedule, valueToScale)
	
end

function BaseValueScheduler:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseValueScheduler:getValueSchedulerInternalParameterArray(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.valueSchedulerInternalParameterArray
		
	else
		
		return self:deepCopyTable(self.valueSchedulerInternalParameterArray)
		
	end
	
end

function BaseValueScheduler:setValueSchedulerInternalParameterArray(valueSchedulerInternalParameterArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.valueSchedulerInternalParameterArray = valueSchedulerInternalParameterArray

	else

		self.valueSchedulerInternalParameterArray = self:deepCopyTable(valueSchedulerInternalParameterArray)

	end

end

function BaseValueScheduler:reset()

	self.valueSchedulerInternalParameterArray = nil

end

return BaseValueScheduler
