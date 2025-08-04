--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
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

function UniformExperienceReplay.new(parameterDictionary)
	
	local NewUniformExperienceReplay = BaseExperienceReplay.new(parameterDictionary)
	
	setmetatable(NewUniformExperienceReplay, UniformExperienceReplay)
	
	NewUniformExperienceReplay:setName("UniformExperienceReplay")
	
	NewUniformExperienceReplay:setRunFunction(function(updateFunction)
		
		local experienceReplayBatchArray = sample(NewUniformExperienceReplay.replayBufferArray, NewUniformExperienceReplay.batchSize)

		for _, experience in ipairs(experienceReplayBatchArray) do updateFunction(table.unpack(experience)) end
		
	end)
	
	return NewUniformExperienceReplay
	
end

return UniformExperienceReplay
