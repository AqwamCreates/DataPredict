BaseExperienceReplay = {}

BaseExperienceReplay.__index = BaseExperienceReplay

local defaultBatchSize = 32

local defaultMaxBufferSize = 100

local defaultNumberOfRunsToUpdate = 1

function BaseExperienceReplay.new(batchSize, numberOfRunsToUpdate, maxBufferSize)
	
	local NewBaseExperienceReplay = {}
	
	setmetatable(NewBaseExperienceReplay, BaseExperienceReplay)

	NewBaseExperienceReplay.batchSize = batchSize or defaultBatchSize

	NewBaseExperienceReplay.numberOfRunsToUpdate = numberOfRunsToUpdate or defaultNumberOfRunsToUpdate

	NewBaseExperienceReplay.maxBufferSize = maxBufferSize or defaultMaxBufferSize
	
	NewBaseExperienceReplay.numberOfRuns = 0
	
	NewBaseExperienceReplay.replayBufferArray = {}
	
	NewBaseExperienceReplay.temporalDifferenceErrorArray = {}
	
	NewBaseExperienceReplay.isTemporalDifferenceErrorRequired = false
	
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

function BaseExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local experience = {previousFeatureVector, action, rewardValue, currentFeatureVector}

	table.insert(self.replayBufferArray, experience)
	
	local addExperienceFunction = self.addExperienceFunction
	
	if (addExperienceFunction) then addExperienceFunction(previousFeatureVector, action, rewardValue, currentFeatureVector) end

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
