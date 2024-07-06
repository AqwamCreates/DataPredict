BaseExperienceReplay = {}

BaseExperienceReplay.__index = BaseExperienceReplay

local defaultBatchSize = 32

local defaultMaxBufferSize = 100

local defaultNumberOfExperienceToUpdate = 1

function BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	local NewBaseExperienceReplay = {}
	
	setmetatable(NewBaseExperienceReplay, BaseExperienceReplay)

	NewBaseExperienceReplay.batchSize = batchSize or defaultBatchSize

	NewBaseExperienceReplay.numberOfExperienceToUpdate = numberOfExperienceToUpdate or defaultNumberOfExperienceToUpdate

	NewBaseExperienceReplay.maxBufferSize = maxBufferSize or defaultMaxBufferSize
	
	NewBaseExperienceReplay.numberOfExperience = 0
	
	NewBaseExperienceReplay.replayBufferArray = {}
	
	NewBaseExperienceReplay.temporalDifferenceErrorArray = {}
	
	NewBaseExperienceReplay.isTemporalDifferenceErrorRequired = false
	
	return NewBaseExperienceReplay
	
end

function BaseExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

function BaseExperienceReplay:extendResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function BaseExperienceReplay:reset()
	
	self.numberOfExperience = 0
	
	table.clear(self.replayBufferArray)
	
	table.clear(self.temporalDifferenceErrorArray)
	
	local resetFunction = self.resetFunction
	
	if resetFunction then resetFunction() end
	
end

function BaseExperienceReplay:setRunFunction(runFunction)
	
	self.runFunction = runFunction
	
end

function BaseExperienceReplay:run(updateFunction)
	
	if (self.numberOfExperience < self.numberOfExperienceToUpdate) then return nil end
	
	self.numberOfExperience = 0
	
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
	
	self.numberOfExperience += 1
	
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
