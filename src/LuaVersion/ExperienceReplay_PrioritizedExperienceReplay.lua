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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local BaseExperienceReplay = require("Model_BaseExperienceReplay")

PrioritizedExperienceReplay = {}

PrioritizedExperienceReplay.__index = PrioritizedExperienceReplay

setmetatable(PrioritizedExperienceReplay, BaseExperienceReplay)

local defaultAlpha = 0.6

local defaultBeta = 0.4

local defaultAggregateFunction = "Maximum"

local defaultEpsilon = math.pow(10, -4)

local aggregrateFunctionList = {
	
	["Maximum"] = function (vector) 
		
		return AqwamMatrixLibrary:findMaximumValue(vector) 
		
	end,
	
	["Minimum"] = function (vector) 

		return AqwamMatrixLibrary:findMinimumValue(vector) 

	end,
	
	["Sum"] = function (vector) 
		
		return AqwamMatrixLibrary:sum(vector) 
		
	end,
	
	["Average"] = function (vector) 

		return AqwamMatrixLibrary:sum(vector) / #vector[1] 

	end,
	
}

local function sample(probabilityArray)
	
	local randomProbability = math.random()
	
	local cumulativeProbability = 0
	
	for i = #probabilityArray, 1, -1 do
		
		local probability = probabilityArray[i]
		
		cumulativeProbability = cumulativeProbability + probability

		if (randomProbability >= cumulativeProbability) then continue end

		return i, probability
		
	end
	
end

function PrioritizedExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, alpha, beta, aggregateFunction, epsilon)
	
	local NewPrioritizedExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewPrioritizedExperienceReplay, PrioritizedExperienceReplay)
	
	NewPrioritizedExperienceReplay.alpha = alpha or defaultAlpha
	
	NewPrioritizedExperienceReplay.beta = beta or defaultBeta
	
	NewPrioritizedExperienceReplay.aggregateFunction = aggregateFunction or defaultAggregateFunction
	
	NewPrioritizedExperienceReplay.epsilon = epsilon or defaultEpsilon
	
	NewPrioritizedExperienceReplay.Model = nil
	
	NewPrioritizedExperienceReplay.priorityArray = {} -- Store priorities
	
	NewPrioritizedExperienceReplay.weightArray = {}
	
	NewPrioritizedExperienceReplay:setIsTemporalDifferenceErrorRequired(true)
	
	NewPrioritizedExperienceReplay:extendAddExperienceFunction(function()
		
		local maxPriority = 1
		
		local priorityArray = NewPrioritizedExperienceReplay.priorityArray
		
		local weightArray = NewPrioritizedExperienceReplay.weightArray
		
		for i, priority in ipairs(priorityArray) do
			
			if (priority <= maxPriority) then continue end
			
			maxPriority = priority
			
		end
		
		table.insert(priorityArray, maxPriority)
		
		table.insert(weightArray, 0)
		
		NewPrioritizedExperienceReplay:removeFirstValueFromArrayIfExceedsBufferSize(priorityArray)
		
		NewPrioritizedExperienceReplay:removeFirstValueFromArrayIfExceedsBufferSize(weightArray)
	
	end)
	
	NewPrioritizedExperienceReplay:extendResetFunction(function()
		
		table.clear(NewPrioritizedExperienceReplay.priorityArray)
		
		table.clear(NewPrioritizedExperienceReplay.weightArray)
		
	end)
	
	NewPrioritizedExperienceReplay:setRunFunction(function()
		
		local Model = NewPrioritizedExperienceReplay.Model

		if (not Model) then error("No Model!") end

		local batchArray = {}

		local alpha = NewPrioritizedExperienceReplay.alpha

		local beta = NewPrioritizedExperienceReplay.beta
		
		local epsilon = NewPrioritizedExperienceReplay.epsilon

		local replayBufferArray = NewPrioritizedExperienceReplay.replayBufferArray
		
		local temporalDifferenceArray = NewPrioritizedExperienceReplay.temporalDifferenceErrorArray
		
		local priorityArray = NewPrioritizedExperienceReplay.priorityArray
		
		local weightArray = NewPrioritizedExperienceReplay.weightArray

		local aggregateFunctionToApply = aggregrateFunctionList[NewPrioritizedExperienceReplay.aggregateFunction]
		
		local batchSize = NewPrioritizedExperienceReplay.batchSize
		
		local replayBufferArraySize = #replayBufferArray

		local lowestNumberOfBatchSize = math.min(batchSize, replayBufferArraySize)		
		
		local probabilityArray = {}

		local sumPriorityAlpha = 0
		
		for i, priority in ipairs(priorityArray) do
			
			local priorityAlpha = math.pow(priority, alpha)
			
			probabilityArray[i] = priorityAlpha
			
			sumPriorityAlpha = sumPriorityAlpha + priorityAlpha
			
		end
		
		for i, probability in ipairs(probabilityArray) do
			
			probabilityArray[i] = probability / sumPriorityAlpha
			
		end

		local sizeArray = AqwamMatrixLibrary:getSize(replayBufferArray[1][1])

		local inputMatrix = AqwamMatrixLibrary:createMatrix(sizeArray[1], sizeArray[2], 1)

		local sumLossMatrix
		
		for i = 1, lowestNumberOfBatchSize, 1 do
			
			local index, probability = sample(probabilityArray, sumPriorityAlpha)
			
			local experience = replayBufferArray[index]
			
			local temporalDifferenceErrorValueOrVector = temporalDifferenceArray[index]

			local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * probability), -beta) / math.max(table.unpack(weightArray), epsilon) 
			
			if (type(temporalDifferenceErrorValueOrVector) ~= "number") then

				temporalDifferenceErrorValueOrVector = aggregateFunctionToApply(temporalDifferenceErrorValueOrVector)

			end
			
			weightArray[index] = importanceSamplingWeight

			priorityArray[index] = math.abs(temporalDifferenceErrorValueOrVector)

			local outputMatrix = Model:forwardPropagate(replayBufferArray[i][1], false)

			local lossMatrix = AqwamMatrixLibrary:multiply(outputMatrix, temporalDifferenceErrorValueOrVector, importanceSamplingWeight)

			if (sumLossMatrix) then

				sumLossMatrix = AqwamMatrixLibrary:add(sumLossMatrix, lossMatrix)

			else

				sumLossMatrix = lossMatrix

			end

		end

		Model:forwardPropagate(inputMatrix, true)

		Model:backPropagate(sumLossMatrix, true)
		
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

return PrioritizedExperienceReplay
