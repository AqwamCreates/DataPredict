local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

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

function PrioritizedExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize, alpha, beta, aggregateFunction, epsilon)
	
	local NewPrioritizedExperienceReplay = BaseExperienceReplay.new(batchSize, numberOfExperienceToUpdate, maxBufferSize)
	
	setmetatable(NewPrioritizedExperienceReplay, PrioritizedExperienceReplay)
	
	NewPrioritizedExperienceReplay.alpha = alpha or defaultAlpha
	
	NewPrioritizedExperienceReplay.beta = beta or defaultBeta
	
	NewPrioritizedExperienceReplay.aggregateFunction = aggregateFunction or defaultAggregateFunction
	
	NewPrioritizedExperienceReplay.epsilon = epsilon or defaultEpsilon
	
	NewPrioritizedExperienceReplay.Model = nil
	
	NewPrioritizedExperienceReplay.sumPriorityAlpha = 0

	NewPrioritizedExperienceReplay.maxPriority = 1

	NewPrioritizedExperienceReplay.maxWeight = 1
	
	NewPrioritizedExperienceReplay.priorityArray = {} -- Store priorities
	
	NewPrioritizedExperienceReplay.probabilityArray = {}
	
	NewPrioritizedExperienceReplay.weightArray = {}
	
	NewPrioritizedExperienceReplay.indexArray = {}
	
	for i = 1, NewPrioritizedExperienceReplay.maxBufferSize, 1 do
		
		table.insert(NewPrioritizedExperienceReplay.priorityArray, NewPrioritizedExperienceReplay.epsilon)
		
		table.insert(NewPrioritizedExperienceReplay.probabilityArray, 0)
		
		table.insert(NewPrioritizedExperienceReplay.weightArray, 0)
		
		table.insert(NewPrioritizedExperienceReplay.indexArray, i)
		
	end
	
	NewPrioritizedExperienceReplay:setIsTemporalDifferenceErrorRequired(true)
	
	NewPrioritizedExperienceReplay:extendAddExperienceFunction(function()
		
		local numberOfExperience = NewPrioritizedExperienceReplay.numberOfExperience
		
		local index = numberOfExperience % NewPrioritizedExperienceReplay.maxBufferSize
		
		local priorityArray = NewPrioritizedExperienceReplay.priorityArray
		
		local weightArray = NewPrioritizedExperienceReplay.weightArray
		
		local sumPriorityAlpha = NewPrioritizedExperienceReplay.sumPriorityAlpha
		
		local maxPriority = NewPrioritizedExperienceReplay.maxPriority
		
		if (numberOfExperience >= NewPrioritizedExperienceReplay.numberOfExperienceToUpdate) then
			
			print("test")
			
			local probability = NewPrioritizedExperienceReplay.probabilityArray[index]
			
			sumPriorityAlpha -= math.pow(probability, NewPrioritizedExperienceReplay.alpha)
			
			if priorityArray[index] == maxPriority then
				
				priorityArray[index] = 0
				
				maxPriority = math.max(table.unpack(priorityArray))
				
			end
			
			if (weightArray[index] == NewPrioritizedExperienceReplay.maxWeight) then

				weightArray[index] = 0

				weightArray = math.max(table.unpack(weightArray))

			end
			
		end
		
		local priority = maxPriority

		local weight = NewPrioritizedExperienceReplay.maxWeight

		sumPriorityAlpha += math.pow(priority, NewPrioritizedExperienceReplay.alpha)

		local probability = math.pow(priority, (NewPrioritizedExperienceReplay.alpha / sumPriorityAlpha))
		
		NewPrioritizedExperienceReplay.priorityArray[index] = priority -- Store priorities

		NewPrioritizedExperienceReplay.probabilityArray[index] = probability

		NewPrioritizedExperienceReplay.weightArray[index] = weight

		NewPrioritizedExperienceReplay.indexArray[index] = index
		
		NewPrioritizedExperienceReplay.sumPriorityAlpha = sumPriorityAlpha
		
		NewPrioritizedExperienceReplay.maxPriority = maxPriority
		
	end)
	
	NewPrioritizedExperienceReplay:extendResetFunction(function()
		
		NewPrioritizedExperienceReplay.sumPriorityAlpha = 0

		NewPrioritizedExperienceReplay.maxPriority = 1

		NewPrioritizedExperienceReplay.maxWeight = 1
		
		table.clear(NewPrioritizedExperienceReplay.priorityArray)
		
		table.clear(NewPrioritizedExperienceReplay.probabilityArray)
		
		table.clear(NewPrioritizedExperienceReplay.weightArray)
		
		table.clear(NewPrioritizedExperienceReplay.weightArray)
		
		for i = 1, NewPrioritizedExperienceReplay.maxBufferSize, 1 do

			table.insert(NewPrioritizedExperienceReplay.priorityArray, NewPrioritizedExperienceReplay.epsilon)
			
			table.insert(NewPrioritizedExperienceReplay.probabilityArray, 0)
			
			table.insert(NewPrioritizedExperienceReplay.weightArray, 0)
			
			table.insert(NewPrioritizedExperienceReplay.indexArray, i)

		end
		
	end)
	
	NewPrioritizedExperienceReplay:setRunFunction(function()
		
		local Model = NewPrioritizedExperienceReplay.Model

		if (not Model) then error("No Model!") end

		local batchArray = {}

		local alpha = NewPrioritizedExperienceReplay.alpha

		local beta = NewPrioritizedExperienceReplay.beta

		local replayBufferArray = NewPrioritizedExperienceReplay.replayBufferArray

		local aggregateFunctionToApply = aggregrateFunctionList[NewPrioritizedExperienceReplay.aggregateFunction]

		local lowestNumberOfBatchSize = math.min(NewPrioritizedExperienceReplay.batchSize, #replayBufferArray)

		local epsilon = NewPrioritizedExperienceReplay.epsilon

		local probabilityArray = NewPrioritizedExperienceReplay.probabilityArray

		local priorityArray = NewPrioritizedExperienceReplay.priorityArray

		local weightArray = NewPrioritizedExperienceReplay.weightArray

		local indexArray = NewPrioritizedExperienceReplay.indexArray

		local maxWeight = NewPrioritizedExperienceReplay.maxWeight

		local sumPriorityAlpha = 0

		for i = 1, #priorityArray, 1 do

			local priority = priorityArray[i]

			sumPriorityAlpha += math.pow(priority, alpha)

		end

		for i = 1, #priorityArray, 1 do

			local probability = math.pow(priorityArray[i], alpha) / sumPriorityAlpha

			local weightPart1 = lowestNumberOfBatchSize * probabilityArray[i]

			local weight = math.pow(weightPart1, -beta) / maxWeight

			probabilityArray[i] = probability

			weightArray[i] = weight

		end

		local sizeArray = AqwamMatrixLibrary:getSize(replayBufferArray[1][1])

		local inputMatrix = AqwamMatrixLibrary:createMatrix(sizeArray[1], sizeArray[2], 1)

		local sumLossMatrix

		for i, temporalDifferenceErrorValueOrVector in ipairs(NewPrioritizedExperienceReplay.temporalDifferenceErrorArray) do

			if (type(temporalDifferenceErrorValueOrVector) ~= "number") then

				temporalDifferenceErrorValueOrVector = aggregateFunctionToApply(temporalDifferenceErrorValueOrVector)

			end

			priorityArray[i] = math.abs(temporalDifferenceErrorValueOrVector)

			local transitionProbability = math.pow(temporalDifferenceErrorValueOrVector, alpha) / sumPriorityAlpha

			if (transitionProbability ~= transitionProbability) then continue end -- Not sure why there is a bug when calculating transition probability for a number of cases. Anyways, it gives nan value when this happens.

			local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * transitionProbability), -beta) / maxWeight

			local outputMatrix = Model:forwardPropagate(replayBufferArray[i][1], false)

			local lossMatrix = AqwamMatrixLibrary:multiply(outputMatrix, temporalDifferenceErrorValueOrVector, importanceSamplingWeight)

			if (sumLossMatrix) then

				sumLossMatrix = AqwamMatrixLibrary:add(sumLossMatrix, lossMatrix)

			else

				sumLossMatrix = lossMatrix

			end

		end

		NewPrioritizedExperienceReplay.sumPriorityAlpha = sumPriorityAlpha

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
