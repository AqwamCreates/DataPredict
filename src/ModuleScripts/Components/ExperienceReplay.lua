ExperienceReplayComponent = {}

ExperienceReplayComponent.__index = ExperienceReplayComponent

local defaultBatchSize = 32

local defaultMaxBufferSize = 100

local defaultNumberOfExperienceToUpdate = 1

function ExperienceReplayComponent.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	local NewExperienceReplayComponent = {}
	
	setmetatable(NewExperienceReplayComponent, ExperienceReplayComponent)

	NewExperienceReplayComponent.batchSize = batchSize or defaultBatchSize

	NewExperienceReplayComponent.numberOfExperienceToUpdate = numberOfExperienceToUpdate or defaultNumberOfExperienceToUpdate

	NewExperienceReplayComponent.maxBufferSize = maxBufferSize or defaultMaxBufferSize
	
	NewExperienceReplayComponent.numberOfExperience = 0
	
	NewExperienceReplayComponent.replayBufferArray = {}
	
	return NewExperienceReplayComponent
	
end

function ExperienceReplayComponent:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

function ExperienceReplayComponent:reset()
	
	self.numberOfExperience = 0

	self.replayBufferArray = {}
	
end

function ExperienceReplayComponent:sampleBatch()
	
	local batchArray = {}
	
	local lowestNumberOfBatchSize = math.min(self.batchSize, #self.replayBufferArray)

	for i = 1, lowestNumberOfBatchSize, 1 do

		local index = Random.new():NextInteger(1, #self.replayBufferArray)

		table.insert(batchArray, self.replayBufferArray[index])

	end

	return batchArray
	
end

function ExperienceReplayComponent:run(updateFunction)
	
	if (self.numberOfExperience < self.numberOfExperienceToUpdate) then return nil end

	local experienceReplayBatchArray = self:sampleBatch()

	for _, experience in ipairs(experienceReplayBatchArray) do -- (s1, a, r, s2)
		
		local previousStateVector = experience[1]
		
		local action = experience[2]
		
		local rewardValue = experience[3]
		
		local currentStateVector = experience[4]

		updateFunction(previousStateVector, action, rewardValue, currentStateVector)

	end
	
	self.numberOfExperience = 0
	
end

function ExperienceReplayComponent:addExperience(previousStateVector, action, rewardValue, currentStateVector)
	
	local experience = {previousStateVector, action, rewardValue, currentStateVector}

	table.insert(self.replayBufferArray, experience)

	if (#self.replayBufferArray > self.maxBufferSize) then table.remove(self.replayBufferArray, 1) end
	
	self.numberOfExperience += 1
	
end

return ExperienceReplayComponent
