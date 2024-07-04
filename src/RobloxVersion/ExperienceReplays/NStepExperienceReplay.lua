local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

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

function NStepExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, nStep, discountFactor)

	local NewNStepExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)

	setmetatable(NewNStepExperienceReplay, NStepExperienceReplay)

	NewNStepExperienceReplay.nStep = nStep or defaultNStep
	
	NewNStepExperienceReplay.discountFactor = discountFactor or defaultDiscountFactor

	NewNStepExperienceReplay:setRunFunction(function(updateFunction)

		local nStep = NewNStepExperienceReplay.nStep
		
		local discountFactor = NewNStepExperienceReplay.discountFactor
		
		local replayBufferArray = NewNStepExperienceReplay.replayBufferArray

		local experienceReplayBatchArray = sample(replayBufferArray, NewNStepExperienceReplay.batchSize)
		
		local firstExperience = experienceReplayBatchArray[1]
		
		local previousState = firstExperience[1]
		
		local nStepReward = discountFactor * firstExperience[3]
		
		nStep = math.min(nStep, #replayBufferArray)

		for i = 2, nStep, 1 do
			
			local experience = experienceReplayBatchArray[i]

			local action = experience[2]
			
			local reward = experience[3]
			
			local currentState = experience[4]
			
			nStepReward = nStepReward + (math.pow(discountFactor, i) * reward)
			
			updateFunction(previousState, action, nStepReward, currentState)
			
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
