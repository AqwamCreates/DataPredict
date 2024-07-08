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

function NStepExperienceReplay.new(batchSize, numberOfRunsToUpdate, maxBufferSize, nStep)

	local NewNStepExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfRunsToUpdate, maxBufferSize)

	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)

	NewNStepExperienceReplay.nStep = nStep or defaultNStep

	NewNStepExperienceReplay:setRunFunction(function(updateFunction)
		
		local discountFactor = NewNStepExperienceReplay.discountFactor
		
		local replayBufferArray = NewNStepExperienceReplay.replayBufferArray

		local replayBufferBatchArray = sample(replayBufferArray, NewNStepExperienceReplay.batchSize)
		
		local replayBufferArraySize = #replayBufferArray
		
		local replayBufferBatchArraySize = #replayBufferBatchArray
		
		local nStepReward = 0
		
		local nStep = math.min(NewNStepExperienceReplay.nStep, replayBufferBatchArraySize)

		for i = replayBufferBatchArraySize, (replayBufferBatchArraySize - nStep), -1 do
			
			local experience = replayBufferBatchArray[i] 

			nStepReward = nStepReward + experience[3]
			
			updateFunction(experience[1], experience[2], nStepReward, experience[4])
			
		end

	end)

	return NewNStepExperienceReplay

end

function NStepExperienceReplay:setParameters(batchSize, numberOfRunsToUpdate, maxBufferSize, nStep)

	self.batchSize = batchSize or self.batchSize

	self.numberOfRunsToUpdate = numberOfRunsToUpdate or self.numberOfRunsToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize

	self.nStep = nStep or self.nStep

end

return NStepExperienceReplay
