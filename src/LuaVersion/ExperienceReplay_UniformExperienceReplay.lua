local BaseExperienceReplay = require("ExperienceReplay_BaseExperienceReplay")

UniformExperienceReplay = {}

UniformExperienceReplay.__index = UniformExperienceReplay

setmetatable(UniformExperienceReplay, BaseExperienceReplay)

function UniformExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	local NewUniformExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewUniformExperienceReplay, UniformExperienceReplay)
	
	NewUniformExperienceReplay:setSampleFunction(function()
		
		local batchArray = {}

		local lowestNumberOfBatchSize = math.min(NewUniformExperienceReplay.batchSize, #NewUniformExperienceReplay.replayBufferArray)

		for i = 1, lowestNumberOfBatchSize, 1 do

			local index = Random.new():NextInteger(1, #NewUniformExperienceReplay.replayBufferArray)

			table.insert(batchArray, NewUniformExperienceReplay.replayBufferArray[index])

		end

		return batchArray
		
	end)
	
	return NewUniformExperienceReplay
	
end

function UniformExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

return UniformExperienceReplay
