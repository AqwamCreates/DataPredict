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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseExperienceReplay = require(script.Parent.BaseExperienceReplay)

PrioritizedExperienceReplay = {}

PrioritizedExperienceReplay.__index = PrioritizedExperienceReplay

setmetatable(PrioritizedExperienceReplay, BaseExperienceReplay)

local defaultAlpha = 0.6

local defaultBeta = 0.4

local defaultAggregateFunction = "Maximum"

local defaultEpsilon = 1e-16

local aggregateFunctionList = {
	
	["Maximum"] = function (valueVector) 
		
		return AqwamTensorLibrary:findMaximumValue(valueVector) 
		
	end,
	
	["Minimum"] = function (valueVector) 

		return AqwamTensorLibrary:findMinimumValue(valueVector) 

	end,
	
	["Sum"] = function (valueVector) 
		
		return AqwamTensorLibrary:sum(valueVector) 
		
	end,
	
	["Average"] = function (valueVector) 

		return AqwamTensorLibrary:sum(valueVector) / #valueVector[1] 

	end,
	
}

local function sample(probabilityArray)
	
	local sumProbability = 0
	
	for i, probability in ipairs(probabilityArray) do
		
		sumProbability = sumProbability + probability
		
	end
	
	local randomProbability = math.random() * sumProbability
	
	local cumulativeProbability = 0
	
	for probabilityIndex, probability in ipairs(probabilityArray) do
		
		cumulativeProbability = cumulativeProbability + probability
		
		if (randomProbability <= cumulativeProbability) then return probabilityIndex, probability end
		
	end
	
	return #probabilityArray, probabilityArray[#probabilityArray]
	
end

function PrioritizedExperienceReplay.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewPrioritizedExperienceReplay = BaseExperienceReplay.new(parameterDictionary)
	
	setmetatable(NewPrioritizedExperienceReplay, PrioritizedExperienceReplay)
	
	NewPrioritizedExperienceReplay:setName("PrioritizedExperienceReplay")
	
	NewPrioritizedExperienceReplay.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewPrioritizedExperienceReplay.beta = parameterDictionary.beta or defaultBeta
	
	NewPrioritizedExperienceReplay.aggregateFunction = parameterDictionary.aggregateFunction or defaultAggregateFunction
	
	NewPrioritizedExperienceReplay.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewPrioritizedExperienceReplay.Model = parameterDictionary.Model
	
	NewPrioritizedExperienceReplay.priorityArray = parameterDictionary.priorityArray or {}
	
	NewPrioritizedExperienceReplay.weightArray = parameterDictionary.weightArray or {}
	
	NewPrioritizedExperienceReplay:setIsTemporalDifferenceErrorRequired(true)
	
	NewPrioritizedExperienceReplay:extendAddExperienceFunction(function()
		
		local maximumPriority = 1
		
		local priorityArray = NewPrioritizedExperienceReplay.priorityArray
		
		local weightArray = NewPrioritizedExperienceReplay.weightArray
		
		for i, priority in ipairs(priorityArray) do
			
			if (priority > maximumPriority) then
				
				maximumPriority = priority
				
			end
			
		end
		
		table.insert(priorityArray, maximumPriority)
		
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

		if (not Model) then error("No model.") end

		local batchArray = {}

		local alpha = NewPrioritizedExperienceReplay.alpha

		local beta = NewPrioritizedExperienceReplay.beta
		
		local epsilon = NewPrioritizedExperienceReplay.epsilon

		local replayBufferArray = NewPrioritizedExperienceReplay.replayBufferArray
		
		local temporalDifferenceArray = NewPrioritizedExperienceReplay.temporalDifferenceErrorArray
		
		local priorityArray = NewPrioritizedExperienceReplay.priorityArray
		
		local weightArray = NewPrioritizedExperienceReplay.weightArray

		local aggregateFunctionToApply = aggregateFunctionList[NewPrioritizedExperienceReplay.aggregateFunction]
		
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
		
		for i = 1, lowestNumberOfBatchSize, 1 do
			
			local index, probability = sample(probabilityArray, sumPriorityAlpha)
			
			local previousFeatureVector = replayBufferArray[index][1]
			
			local temporalDifferenceErrorValueOrVector = temporalDifferenceArray[index]

			local importanceSamplingWeight = math.pow((lowestNumberOfBatchSize * probability), -beta) / math.max(table.unpack(weightArray), epsilon) 
			
			if (type(temporalDifferenceErrorValueOrVector) ~= "number") then

				temporalDifferenceErrorValueOrVector = aggregateFunctionToApply(temporalDifferenceErrorValueOrVector)

			end
			
			weightArray[index] = importanceSamplingWeight

			priorityArray[index] = math.abs(temporalDifferenceErrorValueOrVector)

			local outputVector = Model:forwardPropagate(previousFeatureVector, true)

			local lossMatrix = AqwamTensorLibrary:multiply(outputVector, temporalDifferenceErrorValueOrVector, importanceSamplingWeight)

			Model:update(lossMatrix, true)

		end

	end)
	
	return NewPrioritizedExperienceReplay
	
end

function PrioritizedExperienceReplay:setModel(Model)
	
	self.Model = Model or self.Model
	
end

return PrioritizedExperienceReplay
