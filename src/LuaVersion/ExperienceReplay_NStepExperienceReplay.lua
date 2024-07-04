--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
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

local defaultDiscountFactor = 0.95

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

		local experienceReplayBatchArray = sample(replayBufferArray, NewNStepExperienceReplay.batchSize)
		
		local replayBatchArraySize = #replayBufferArray
		
		local firstExperience = experienceReplayBatchArray[1]
		
		local currentState = firstExperience[1]
		
		local action = firstExperience[2]
		
		local nStepReward = firstExperience[3]
		
		local nStep = math.min(NewNStepExperienceReplay.nStep, replayBatchArraySize - 1)

		for i = 1, nStep, 1 do
			
			local experience = experienceReplayBatchArray[i + 1]
			
			local previousState = experience[1]
			
			local reward = experience[3]
			
			nStepReward = nStepReward + reward
			
			updateFunction(previousState, action, nStepReward, currentState)
			
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
