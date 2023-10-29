local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

PrioritizedExperienceReplay = {}

PrioritizedExperienceReplay.__index = PrioritizedExperienceReplay

setmetatable(PrioritizedExperienceReplay, BaseExperienceReplay)

function PrioritizedExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	local NewPrioritizedExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewPrioritizedExperienceReplay, PrioritizedExperienceReplay)
	
	NewPrioritizedExperienceReplay.priorities = {} -- Store priorities
	
	NewPrioritizedExperienceReplay:setSampleFunction(function()
		
		local batchArray = {}
		
		local prioritySum = 0

		for i, priority in ipairs(NewPrioritizedExperienceReplay.priorities) do prioritySum += priority end

		for _ = 1, NewPrioritizedExperienceReplay.batchSize, 1 do
			
			local threshold = prioritySum * math.random()
			
			local cumulativePriority = 0

			for i, priority in ipairs(NewPrioritizedExperienceReplay.priorities) do
				
				cumulativePriority += priority

				if cumulativePriority < threshold then continue end
				
				table.insert(batchArray, NewPrioritizedExperienceReplay.replayBufferArray[i])
				
				break
				
			end
			
		end
		
		return batchArray
		
	end)
	
	NewPrioritizedExperienceReplay:setResetFunction(function()
		
		NewPrioritizedExperienceReplay.numberOfExperience = 0

		NewPrioritizedExperienceReplay.replayBufferArray = {}
		
		NewPrioritizedExperienceReplay.priorities = {}
		
	end)
	
	return NewPrioritizedExperienceReplay
	
end

function PrioritizedExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

return PrioritizedExperienceReplay
