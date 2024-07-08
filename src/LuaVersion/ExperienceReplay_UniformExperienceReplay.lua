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

UniformExperienceReplay = {}

UniformExperienceReplay.__index = UniformExperienceReplay

setmetatable(UniformExperienceReplay, BaseExperienceReplay)

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

function UniformExperienceReplay.new(batchSize, numberOfRunsToUpdate, maxBufferSize)
	
	local NewUniformExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfRunsToUpdate, maxBufferSize)
	
	setmetatable(NewUniformExperienceReplay, UniformExperienceReplay)
	
	NewUniformExperienceReplay:setRunFunction(function(updateFunction)
		
		local experienceReplayBatchArray = sample(NewUniformExperienceReplay.replayBufferArray, NewUniformExperienceReplay.batchSize)

		for _, experience in ipairs(experienceReplayBatchArray) do -- (s1, a, r, s2)

			local previousFeatureVector = experience[1]

			local action = experience[2]

			local rewardValue = experience[3]

			local currentFeatureVector = experience[4]

			updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

		end
		
	end)
	
	return NewUniformExperienceReplay
	
end

function UniformExperienceReplay:setParameters(batchSize, numberOfRunsToUpdate, maxBufferSize)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfRunsToUpdate = numberOfRunsToUpdate or self.numberOfRunsToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
end

return UniformExperienceReplay
