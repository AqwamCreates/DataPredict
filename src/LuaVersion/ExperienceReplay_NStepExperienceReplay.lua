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

		local index = Random.new():NextInteger(1, replayBufferArraySize)

		table.insert(batchArray, replayBufferArray[index])

	end

	return batchArray

end

function NStepExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep, discountFactor)

	local NewNStepExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)

	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)

	NewNStepExperienceReplay.nStep = nStep or defaultNStep
	
	NewNStepExperienceReplay.discountFactor = discountFactor or defaultDiscountFactor

	NewNStepExperienceReplay:setRunFunction(function(updateFunction)

		local nStep = NewNStepExperienceReplay.nStep
		
		local discountFactor = NewNStepExperienceReplay.discountFactor

		local experienceReplayBatchArray = sample(NewNStepExperienceReplay.replayBufferArray, NewNStepExperienceReplay.batchSize)

		for experienceIndex, experience in ipairs(experienceReplayBatchArray) do

			local nStepRewards = 0

			local previousState = experience[1]

			local action = experience[2]
			
			local currentState = experience[4]

			for i = 1, nStep, 1 do

				if (not experienceReplayBatchArray[i]) then continue end

				nStepRewards += math.pow(discountFactor, i) * experienceReplayBatchArray[i][3]

			end

			updateFunction(previousState, action, nStepRewards, currentState)

		end

	end)

	return NewNStepExperienceReplay

end

function NStepExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep, discountFactor)

	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize

	self.nStep = nStep or self.nStep
	
	self.discountFactor = discountFactor or self.discountFactor

end

return NStepExperienceReplay
