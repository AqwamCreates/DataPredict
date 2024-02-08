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
	
	NewPrioritizedExperienceReplay.priorityArray = {} -- Store priorities
	
	NewPrioritizedExperienceReplay.probabilityArray = {}
	
	NewPrioritizedExperienceReplay.weightArray = {}
	
	NewPrioritizedExperienceReplay.indexArray = {}
	
	NewPrioritizedExperienceReplay.sumPriorityAlpha = 0
	
	NewPrioritizedExperienceReplay.maxPriority = 1
	
	NewPrioritizedExperienceReplay.maxWeight = 1
	
	for i = 1, NewPrioritizedExperienceReplay.maxBufferSize, 1 do
		
		table.insert(NewPrioritizedExperienceReplay.priorityArray, NewPrioritizedExperienceReplay.epsilon)
		table.insert(NewPrioritizedExperienceReplay.probabilityArray, 0)
		table.insert(NewPrioritizedExperienceReplay.weightArray, 0)
		table.insert(NewPrioritizedExperienceReplay.indexArray, i)
		
	end
	
	NewPrioritizedExperienceReplay:setIsTemporalDifferenceErrorRequired(true)
	
	NewPrioritizedExperienceReplay:extendAddExperienceFunction(function(experience)
		
		local numberOfExperience = NewPrioritizedExperienceReplay.numberOfExperience
		
		local index = numberOfExperience % NewPrioritizedExperienceReplay.maxBufferSize
		
		if (numberOfExperience >= NewPrioritizedExperienceReplay.numberOfExperienceToUpdate) then
			
			local probability = NewPrioritizedExperienceReplay.probabilityArray[index]
			
			NewPrioritizedExperienceReplay.sumPriorityAlpha -= math.pow(probability, NewPrioritizedExperienceReplay.alpha)
			
			if NewPrioritizedExperienceReplay.priority[index] == NewPrioritizedExperienceReplay.maxPriority then
				
				NewPrioritizedExperienceReplay.priority[index] = 0
				
				NewPrioritizedExperienceReplay.maxPriority = math.max(table.unpack(NewPrioritizedExperienceReplay.priorityArray))
				
			end
			
			if (NewPrioritizedExperienceReplay.weightArray[index] == NewPrioritizedExperienceReplay.maxWeight) then

				NewPrioritizedExperienceReplay.weightArray[index] = 0

				NewPrioritizedExperienceReplay.weightArray = math.max(table.unpack(NewPrioritizedExperienceReplay.weightArray))

			end
			
		end
		
		local priority = NewPrioritizedExperienceReplay.maxPriority

		local weight = NewPrioritizedExperienceReplay.maxWeight

		NewPrioritizedExperienceReplay.sumPriorityAlpha += math.pow(priority, NewPrioritizedExperienceReplay.alpha)

		local probability = math.pow(priority, (NewPrioritizedExperienceReplay.alpha / NewPrioritizedExperienceReplay.sumPriorityAlpha))
		
		NewPrioritizedExperienceReplay.priorityArray[index] = priority-- Store priorities

		NewPrioritizedExperienceReplay.probabilityArray[index] = probability

		NewPrioritizedExperienceReplay.weightArray[index] = weight

		NewPrioritizedExperienceReplay.indexArray[index] = index
		
	end)
	
	NewPrioritizedExperienceReplay:extendResetFunction(function()
		
		NewPrioritizedExperienceReplay.sumPriorityAlpha = 0

		NewPrioritizedExperienceReplay.maxPriority = 1

		NewPrioritizedExperienceReplay.maxWeight = 1
		
		table.clear(NewPrioritizedExperienceReplay.priorityArray)
		
		table.clear(NewPrioritizedExperienceReplay.probabilityArray)
		
		table.clear(NewPrioritizedExperienceReplay.weightArray)
		
		table.clear(NewPrioritizedExperienceReplay.weightArray)
		
		for i = 1, NewPrioritizedExperienceReplay.indexArray, 1 do

			table.insert(NewPrioritizedExperienceReplay.priorityArray, NewPrioritizedExperienceReplay.epsilon)
			table.insert(NewPrioritizedExperienceReplay.probabilityArray, 0)
			table.insert(NewPrioritizedExperienceReplay.weightArray, 0)
			table.insert(NewPrioritizedExperienceReplay.indexArray, i)

		end
		
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
	
	local epsilon = self.epsilon
	
	self.sumPriorityAlpha = 0
	
	local probabilityArray = self.probabilityArray
	
	local priorityArray = self.priorityArray
	
	local weightArray = self.weightArray
	
	local indexArray = self.indexArray
	
	for i = 1, #priorityArray, 1 do
		
		self.sumPriorityAlpha += math.pow(priorityArray[i], alpha)
		
	end
	
	for i = 1, #priorityArray, 1 do
		
		local probability = math.pow(priorityArray[i], alpha) / self.sumPriorityAlpha
			
		local weightPart1 = lowestNumberOfBatchSize * probabilityArray[i]
			
		local weight = math.pow(weightPart1, -beta) / self.maxWeight
		
		priorityArray[i] = priorityArray[i]

		probabilityArray[i] = probability

		weightArray[i] = weight
		
		indexArray[i] = indexArray[i]
	
	end
	
	local sizeArray = AqwamMatrixLibrary:getSize(replayBufferArray[1][1])

	local inputMatrix = AqwamMatrixLibrary:createMatrix(sizeArray[1], sizeArray[2], 1)
	
	local sumLossMatrix
	
	for i, temporalDifferenceErrorValueOrVector in ipairs(self.temporalDifferenceErrorArray) do

		local temporalDifferenceErrorValue = temporalDifferenceErrorValueOrVector

		if (type(temporalDifferenceErrorValue) ~= "number") then

			temporalDifferenceErrorValue = aggregateFunctionToApply(temporalDifferenceErrorValue)

		end

		local transitionProbability = math.pow(temporalDifferenceErrorValue, alpha) / self.sumPriorityAlpha
		
		if (transitionProbability ~= transitionProbability) then continue end -- Not sure why there is a bug when calculating transition probability for a number of cases. Anyways, it gives nan value when this happens.

		local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * transitionProbability), -beta) / self.maxWeight

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
