ExperienceReplay = {}

ExperienceReplay.__index = ExperienceReplay

local defaultBatchSize = 32

local defaultMaxBufferSize = 100

local defaultNumberOfExperienceToUpdate = 1

function ExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	local NewExperienceReplay = {}
	
	setmetatable(NewExperienceReplay, ExperienceReplay)

	NewExperienceReplay.batchSize = batchSize or defaultBatchSize

	NewExperienceReplay.numberOfExperienceToUpdate = numberOfExperienceToUpdate or defaultNumberOfExperienceToUpdate

	NewExperienceReplay.maxBufferSize = maxBufferSize or defaultMaxBufferSize
	
	NewExperienceReplay.numberOfExperience = 0
	
	NewExperienceReplay.replayBufferArray = {}
	
	return NewExperienceReplay
	
end

function ExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

function ExperienceReplay:reset()
	
	self.numberOfExperience = 0

	self.replayBufferArray = {}
	
end

function ExperienceReplay:sampleBatch()
	
	local batchArray = {}
	
	local lowestNumberOfBatchSize = math.min(self.batchSize, #self.replayBufferArray)

	for i = 1, lowestNumberOfBatchSize, 1 do

		local index = Random.new():NextInteger(1, #self.replayBufferArray)

		table.insert(batchArray, self.replayBufferArray[index])

	end

	return batchArray
	
end

function ExperienceReplay:run(updateFunction)
	
	if (self.numberOfExperience < self.numberOfExperienceToUpdate) then return nil end
	
	self.numberOfExperience = 0

	local experienceReplayBatchArray = self:sampleBatch()

	for _, experience in ipairs(experienceReplayBatchArray) do -- (s1, a, r, s2)
		
		local previousState = experience[1]
		
		local action = experience[2]
		
		local rewardValue = experience[3]
		
		local currentState = experience[4]

		updateFunction(previousState, action, rewardValue, currentState)

	end
	
end

function ExperienceReplay:addExperience(previousState, action, rewardValue, currentState)
	
	local experience = {previousState, action, rewardValue, currentState}

	table.insert(self.replayBufferArray, experience)

	if (#self.replayBufferArray > self.maxBufferSize) then table.remove(self.replayBufferArray, 1) end
	
	self.numberOfExperience += 1
	
end

return ExperienceReplay
