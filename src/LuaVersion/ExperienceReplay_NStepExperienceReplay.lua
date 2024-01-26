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
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseExperienceReplay = require("ExperienceReplay_BaseExperienceReplay")

NStepExperienceReplay = {}

NStepExperienceReplay.__index = NStepExperienceReplay

setmetatable(NStepExperienceReplay, BaseExperienceReplay)

local defaultNStep = 3

function NStepExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep)
	
	local NewNStepExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)
	
	BaseExperienceReplay.nStep = nStep or defaultNStep
	
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

