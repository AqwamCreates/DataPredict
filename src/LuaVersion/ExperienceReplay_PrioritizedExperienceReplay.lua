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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

PrioritizedExperienceReplay = {}

PrioritizedExperienceReplay.__index = PrioritizedExperienceReplay

setmetatable(PrioritizedExperienceReplay, BaseExperienceReplay)

local defaultAlpha = 0.6

local defaultBeta = 0.4

local defaultAggregateFunction = "Maximum"

local defaultEpsilon = math.pow(10, -4)

local aggregrateFunctionList = {
	
	["Maximum"] = function (vector) 
		
		return AqwamMatrixLibrary:findMaximumValueInMatrix(vector) 
		
	end,
	
	["Sum"] = function (vector) 
		
		return AqwamMatrixLibrary:sum(vector) 
		
	end,
	
}

function PrioritizedExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, alpha, beta, aggregateFunction, epsilon)
	
	local NewPrioritizedExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewPrioritizedExperienceReplay, PrioritizedExperienceReplay)
	
	NewPrioritizedExperienceReplay.alpha = alpha or defaultAlpha
	
	NewPrioritizedExperienceReplay.beta = beta or defaultBeta
	
	NewPrioritizedExperienceReplay.aggregateFunction = aggregateFunction or defaultAggregateFunction
	
	NewPrioritizedExperienceReplay.epsilon = epsilon or defaultEpsilon
	
	NewPrioritizedExperienceReplay.Model = nil
	
	NewPrioritizedExperienceReplay.prioritiesArray = {} -- Store priorities
	
	NewPrioritizedExperienceReplay:setIsTemporalDifferenceErrorRequired(true)
	
	NewPrioritizedExperienceReplay:extendResetFunction(function()
		
		table.clear(NewPrioritizedExperienceReplay.prioritiesArray)
		
	end)
	
	return NewPrioritizedExperienceReplay
	
end

function PrioritizedExperienceReplay:setModel(Model)
	
	self.Model = Model or self.Model
	
end

function PrioritizedExperienceReplay:setParameters(batchSize, numberOfExperienceToUpdate, maxBufferSize, alpha, beta, aggregateFunction, epsilon)
	
	self.batchSize = batchSize or self.batchSize

	self.numberOfExperienceToUpdate = numberOfExperienceToUpdate or self.numberOfExperienceToUpdate

	self.maxBufferSize = maxBufferSize or self.maxBufferSize
	
	self.alpha = alpha or self.alpha

	self.beta = beta or self.beta

	self.aggregateFunction = aggregateFunction or self.aggregateFunction
	
	self.epsilon = epsilon or self.epsilon
	
end

function PrioritizedExperienceReplay:run()
	
	if (not self.Model) then error("No Model!") end
	
	if (self.numberOfExperience < self.numberOfExperienceToUpdate) then return nil end
	
	self.numberOfExperience = 0
	
	local batchArray = {}

	local alpha = self.alpha

	local beta = self.beta
	
	local replayBufferArray = self.replayBufferArray

	local aggregateFunctionToApply = aggregrateFunctionList[self.aggregateFunction]

	local lowestNumberOfBatchSize = math.min(self.batchSize, #replayBufferArray)

	local prioritiesArray = self.prioritiesArray
	
	local epsilon = self.epsilon
	
	local totalPrioritySum = epsilon * #prioritiesArray
	
	local sizeArray = AqwamMatrixLibrary:getSize(replayBufferArray[1][1])
	
	local inputMatrix = AqwamMatrixLibrary:createMatrix(sizeArray[1], sizeArray[2], 1)
	
	local sumLossMatrix

	for i, priorityValue in ipairs(prioritiesArray) do totalPrioritySum += math.pow(priorityValue, alpha) end

	for i, temporalDifferenceErrorValueOrVector in ipairs(self.temporalDifferenceErrorArray) do

		local temporalDifferenceErrorValue = temporalDifferenceErrorValueOrVector

		if (type(temporalDifferenceErrorValue) ~= "number") then

			temporalDifferenceErrorValue = aggregateFunctionToApply(temporalDifferenceErrorValue)

		end

		local transitionProbability = math.pow(temporalDifferenceErrorValue, alpha) / totalPrioritySum

		local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * transitionProbability), -beta)

		prioritiesArray[i] = math.abs(temporalDifferenceErrorValue) + epsilon
		
		local outputMatrix = self.Model:forwardPropagate(replayBufferArray[i][1], false)
		
		local adjustedOutputMatrix = AqwamMatrixLibrary:multiply(outputMatrix, temporalDifferenceErrorValue, importanceSamplingWeight)
		
		local lossMatrix = AqwamMatrixLibrary:subtract(adjustedOutputMatrix, outputMatrix)
		
		if (sumLossMatrix) then
			
			sumLossMatrix = AqwamMatrixLibrary:add(sumLossMatrix, lossMatrix)
			
		else
			
			sumLossMatrix = lossMatrix
			
		end
		
	end
	
	self.Model:forwardPropagate(inputMatrix, true)
	
	self.Model:backPropagate(sumLossMatrix, true)
	
end

return PrioritizedExperienceReplay
