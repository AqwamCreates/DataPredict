local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

NStepExperienceReplay = {}

NStepExperienceReplay.__index = NStepExperienceReplay

setmetatable(NStepExperienceReplay, BaseExperienceReplay)

local defaultNStep = 3

local function sample(replayBufferArray, batchSize)

	local batchArray = {}

	local replayBufferArray = replayBufferArray

	local replayBufferArraySize = #replayBufferArray

	local lowestNumberOfBatchSize = math.min(batchSize, replayBufferArraySize)

	for i = 1, lowestNumberOfBatchSize, 1 do

		table.insert(batchArray, replayBufferArray[i])

	end

	return batchArray

end

function NStepExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep)

	local NewNStepExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)

	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)

	NewNStepExperienceReplay.nStep = nStep or defaultNStep

	NewNStepExperienceReplay:setRunFunction(function(updateFunction)
		
		local discountFactor = NewNStepExperienceReplay.discountFactor
		
		local replayBufferArray = NewNStepExperienceReplay.replayBufferArray

		local replayBatchArray = sample(replayBufferArray, NewNStepExperienceReplay.batchSize)
		
		local replayBufferArraySize = #replayBufferArray
		
		local replayBatchArraySize = #replayBatchArray
		
		local nStepReward = 0
		
		local nStep = math.min(NewNStepExperienceReplay.nStep, replayBatchArraySize)

		for i = replayBatchArraySize, (replayBatchArraySize - nStep), -1 do
			
			local experience = replayBatchArray[i] 

			nStepReward = nStepReward + experience[3]
			
			updateFunction(experience[1], experience[2], nStepReward, experience[4])
			
		end

	end)

	return NewNStepExperienceReplay

end

function NStepExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep)

	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize

	self.nStep = nStep or self.nStep

end

return NStepExperienceReplay
