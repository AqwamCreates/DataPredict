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

		local lowestNumberOfBatchSize = math.min(NewNStepExperienceReplay.batchSize, #NewNStepExperienceReplay.replayBufferArray)

		for i = 1, lowestNumberOfBatchSize, 1 do

			local index = Random.new():NextInteger(1, #NewNStepExperienceReplay.replayBufferArray)

			table.insert(batchArray, NewNStepExperienceReplay.replayBufferArray[index])

		end

		return batchArray
		
	end)
	
	return NewNStepExperienceReplay
	
end

function NStepExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep)

	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
	self.nStep = nStep or self.nStep

end

function NStepExperienceReplay:run(updateFunction)
	
	if self.numberOfExperience < self.numberOfExperienceToUpdate then return end

	self.numberOfExperience = 0

	local experienceReplayBatchArray = self:sample()

	for _, experience in ipairs(experienceReplayBatchArray) do
		
		local nStepRewards = 0
		
		local currentState = experience[4]
		
		local previousState = experience[1]
		
		local action = experience[2]
		
		local nStep = self.nStep
		
		for i = 1, nStep do
			
			if not experienceReplayBatchArray[i] then break end
			
			nStepRewards += experienceReplayBatchArray[i][3]
			
		end

		updateFunction(previousState, action, nStepRewards, currentState)
		
	end
	
end

return NStepExperienceReplay
