--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseInstance = require("Core_BaseInstance")

BaseExperienceReplay = {}

BaseExperienceReplay.__index = BaseExperienceReplay

setmetatable(BaseExperienceReplay, BaseInstance)

local defaultBatchSize = 32

local defaultMaxBufferSize = 100

local defaultNumberOfRunsToUpdate = 1

local defaultNumberOfRuns = 0

function BaseExperienceReplay.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseExperienceReplay = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewBaseExperienceReplay, BaseExperienceReplay)
	
	NewBaseExperienceReplay:setName("ExperienceReplay")
	
	NewBaseExperienceReplay:setName("BaseExperienceReplay")

	NewBaseExperienceReplay.batchSize = parameterDictionary.batchSize or defaultBatchSize

	NewBaseExperienceReplay.numberOfRunsToUpdate = parameterDictionary.numberOfRunsToUpdate or defaultNumberOfRunsToUpdate

	NewBaseExperienceReplay.maxBufferSize = parameterDictionary.maxBufferSize or defaultMaxBufferSize
	
	NewBaseExperienceReplay.numberOfRuns = parameterDictionary.numberOfRuns or defaultNumberOfRuns
	
	NewBaseExperienceReplay.replayBufferArray = parameterDictionary.replayBufferArray or {}
	
	NewBaseExperienceReplay.temporalDifferenceErrorArray = parameterDictionary.temporalDifferenceErrorArray or {}
	
	NewBaseExperienceReplay.isTemporalDifferenceErrorRequired = NewBaseExperienceReplay:getValueOrDefaultValue(parameterDictionary.isTemporalDifferenceErrorRequired, false) 
	
	return NewBaseExperienceReplay
	
end

function BaseExperienceReplay:setParameters(batchSize, numberOfRunsToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfRunsToUpdate = numberOfRunsToUpdate or self.numberOfRunsToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

function BaseExperienceReplay:extendResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function BaseExperienceReplay:reset()
	
	self.numberOfRuns = 0
	
	table.clear(self.replayBufferArray)
	
	table.clear(self.temporalDifferenceErrorArray)
	
	local resetFunction = self.resetFunction
	
	if resetFunction then resetFunction() end
	
end

function BaseExperienceReplay:setRunFunction(runFunction)
	
	self.runFunction = runFunction
	
end

function BaseExperienceReplay:run(updateFunction)
	
	self.numberOfRuns += 1
	
	if (self.numberOfRuns < self.numberOfRunsToUpdate) then return nil end
	
	self.numberOfRuns = 0
	
	self.runFunction(updateFunction)
	
end

function BaseExperienceReplay:removeFirstValueFromArrayIfExceedsBufferSize(targetArray)
	
	if (#targetArray > self.maxBufferSize) then table.remove(targetArray, 1) end
	
end

function BaseExperienceReplay:extendAddExperienceFunction(addExperienceFunction)
	
	self.addExperienceFunction = addExperienceFunction
	
end

function BaseExperienceReplay:addExperience(...)
	
	local experience = {...}

	table.insert(self.replayBufferArray, experience)
	
	local addExperienceFunction = self.addExperienceFunction
	
	if (addExperienceFunction) then addExperienceFunction(...) end

	self:removeFirstValueFromArrayIfExceedsBufferSize(self.replayBufferArray)
	
end

function BaseExperienceReplay:extendAddTemporalDifferenceErrorFunction(addTemporalDifferenceErrorFunction)
	
	self.addTemporalDifferenceErrorFunction = addTemporalDifferenceErrorFunction
	
end

function BaseExperienceReplay:addTemporalDifferenceError(temporalDifferenceErrorVectorOrValue)
	
	if (not self.isTemporalDifferenceErrorRequired) then return nil end
	
	table.insert(self.temporalDifferenceErrorArray, temporalDifferenceErrorVectorOrValue)
	
	local addTemporalDifferenceErrorFunction = self.addTemporalDifferenceErrorFunction
	
	if (addTemporalDifferenceErrorFunction) then addTemporalDifferenceErrorFunction(temporalDifferenceErrorVectorOrValue) end
	
	self:removeFirstValueFromArrayIfExceedsBufferSize(self.temporalDifferenceErrorArray)
	
end

function BaseExperienceReplay:setIsTemporalDifferenceErrorRequired(option)
	
	self.isTemporalDifferenceErrorRequired = option
	
end

return BaseExperienceReplay