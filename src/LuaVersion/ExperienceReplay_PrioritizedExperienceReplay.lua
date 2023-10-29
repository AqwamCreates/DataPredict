--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseExperienceReplay = require("ExperienceReplay_BaseExperienceReplay")

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
