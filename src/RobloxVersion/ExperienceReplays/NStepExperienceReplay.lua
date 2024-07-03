local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

NStepExperienceReplay = {}

NStepExperienceReplay.__index = NStepExperienceReplay

setmetatable(NStepExperienceReplay, BaseExperienceReplay)

local defaultNStep = 3

function NStepExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep)

	local NewNStepExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)

	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)

	NewNStepExperienceReplay.nStep = nStep or defaultNStep

	NewNStepExperienceReplay:setSampleFunction(function()

		local batchArray = {}

		local replayBufferArray = NewNStepExperienceReplay.replayBufferArray

		local replayBufferArraySize = #replayBufferArray

		local lowestNumberOfBatchSize = math.min(NewNStepExperienceReplay.batchSize, replayBufferArraySize)

		for i = 1, lowestNumberOfBatchSize, 1 do

			local index = Random.new():NextInteger(1, replayBufferArraySize)

			table.insert(batchArray, replayBufferArray[index])

		end

		return batchArray

	end)

	NewNStepExperienceReplay:setRunFunction(function(updateFunction)

		local experienceReplayBatchArray = NewNStepExperienceReplay:sample()
		
		local nStep = NewNStepExperienceReplay.nStep

		for _, experience in ipairs(experienceReplayBatchArray) do

			local nStepRewards = 0

			local previousState = experience[1]

			local action = experience[2]
			
			local currentState = experience[4]

			for i = 1, nStep do

				if (not experienceReplayBatchArray[i]) then continue end

				nStepRewards += experienceReplayBatchArray[i][3]

			end

			updateFunction(previousState, action, nStepRewards, currentState)

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
