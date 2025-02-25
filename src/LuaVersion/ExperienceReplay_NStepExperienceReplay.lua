--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

function NStepExperienceReplay.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewNStepExperienceReplay = BaseExperienceReplay.new(parameterDictionary)

	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)
	
	NewNStepExperienceReplay:setName("NStepExperienceReplay")

	NewNStepExperienceReplay.nStep = parameterDictionary.nStep or defaultNStep

	NewNStepExperienceReplay:setRunFunction(function(updateFunction)
		
		local discountFactor = NewNStepExperienceReplay.discountFactor
		
		local replayBufferArray = NewNStepExperienceReplay.replayBufferArray

		local replayBufferBatchArray = sample(replayBufferArray, NewNStepExperienceReplay.batchSize)
		
		local replayBufferArraySize = #replayBufferArray
		
		local replayBufferBatchArraySize = #replayBufferBatchArray
		
		local nStep = math.min(NewNStepExperienceReplay.nStep, replayBufferBatchArraySize)
		
		local finalBatchArrayIndex = (replayBufferBatchArraySize - nStep) + 1

		for i = replayBufferBatchArraySize, finalBatchArrayIndex, -1 do updateFunction(table.unpack(replayBufferBatchArray[i])) end

	end)

	return NewNStepExperienceReplay

end

return NStepExperienceReplay