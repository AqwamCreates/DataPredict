local ExperienceReplayBase = require(script.Parent.ExperienceReplayBase)

ExperienceReplay = {}

ExperienceReplay.__index = ExperienceReplay

setmetatable(ExperienceReplay, ExperienceReplayBase)

function ExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	local NewExperienceReplay = ExperienceReplayBase.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewExperienceReplay, ExperienceReplay)
	
	NewExperienceReplay:setSampleFunction(function()
		
		local batchArray = {}

		local lowestNumberOfBatchSize = math.min(NewExperienceReplay.batchSize, #NewExperienceReplay.replayBufferArray)

		for i = 1, lowestNumberOfBatchSize, 1 do

			local index = Random.new():NextInteger(1, #NewExperienceReplay.replayBufferArray)

			table.insert(batchArray, NewExperienceReplay.replayBufferArray[index])

		end

		return batchArray
		
	end)
	
	NewExperienceReplay:setResetFunction(function()
		
		NewExperienceReplay.numberOfExperience = 0

		NewExperienceReplay.replayBufferArray = {}
		
	end)
	
	return NewExperienceReplay
	
end

function ExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

return ExperienceReplay
