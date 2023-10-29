local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

HindsightExperienceReplay = {}

HindsightExperienceReplay.__index = HindsightExperienceReplay

setmetatable(HindsightExperienceReplay, BaseExperienceReplay)

local defaultHindsightRewardValue = 1

function HindsightExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, hindsightRewardValue)
	
	local NewHindsightExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewHindsightExperienceReplay, HindsightExperienceReplay)
	
	NewHindsightExperienceReplay.hindsightRewardValue = hindsightRewardValue or defaultHindsightRewardValue
	
	NewHindsightExperienceReplay:setSampleFunction(function()
		
		local batchArray = {}

		for i = 1, NewHindsightExperienceReplay.batchSize do
			
			local index = Random.new():NextInteger(1, #NewHindsightExperienceReplay.replayBufferArray)
			
			table.insert(batchArray, NewHindsightExperienceReplay.replayBufferArray[index])
			
		end

		return batchArray
		
	end)
	
	NewHindsightExperienceReplay:setResetFunction(function()
		
		NewHindsightExperienceReplay.numberOfExperience = 0

		NewHindsightExperienceReplay.replayBufferArray = {}
		
	end)
	
	return NewHindsightExperienceReplay
	
end

function HindsightExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

function HindsightExperienceReplay:run(updateFunction)
	
	if self.numberOfExperience < self.numberOfExperienceToUpdate then return nil end

	self.numberOfExperience = 0

	local experienceReplayBatchArray = self:sample()

	for _, experience in ipairs(experienceReplayBatchArray) do
		
		local previousState = experience[1]
		
		local action = experience[2]
		
		local rewardValue = experience[3]
		
		local currentState = experience[4]

		updateFunction(previousState, action, rewardValue, currentState)

		local achievedGoalState = experience[5]
		
		if not achievedGoalState then continue end
			
		local hindsightExperience = {previousState, action, self.hindsightRewardValue, achievedGoalState}
			
		table.insert(self.replayBufferArray, hindsightExperience)
		
	end
	
end

return HindsightExperienceReplay
