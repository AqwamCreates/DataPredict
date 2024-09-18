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

BaseOptimizer = {}

BaseOptimizer.__index = BaseOptimizer

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

end


function BaseOptimizer.new(optimizerName)
	
	local NewBaseOptimizer = {}
	
	setmetatable(NewBaseOptimizer, BaseOptimizer)
	
	NewBaseOptimizer.optimizerName = optimizerName or "Unknown"
	
	NewBaseOptimizer.LearningRateValueScheduler = nil

	NewBaseOptimizer.calculateFunction = nil
	
	NewBaseOptimizer.optimizerInternalParameters = nil
	
	return NewBaseOptimizer
	
end

function BaseOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	local calculateFunction = self.calculateFunction
	
	local LearningRateValueScheduler = self.LearningRateValueScheduler
	
	if (not calculateFunction) then error("No calculate function for the optimizer!") end
	
	if LearningRateValueScheduler then learningRate = LearningRateValueScheduler:calculate(learningRate) end
	
	return self.calculateFunction(learningRate, costFunctionDerivatives)
	
end

function BaseOptimizer:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseOptimizer:setLearningRateValueScheduler(LearningRateValueScheduler)
	
	self.LearningRateValueScheduler = LearningRateValueScheduler
	
end

function BaseOptimizer:getOptimizerName()
	
	return self.optimizerName
	
end

function BaseOptimizer:getOptimizerInternalParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.optimizerInternalParameters
		
	else
		
		return deepCopyTable(self.optimizerInternalParameters)
		
	end
	
end

function BaseOptimizer:getLearningRateValueScheduler()
	
	return self.LearningRateValueScheduler
	
end

function BaseOptimizer:setOptimizerInternalParameters(optimizerInternalParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.optimizerInternalParameters = optimizerInternalParameters

	else

		self.optimizerInternalParameters = deepCopyTable(optimizerInternalParameters)

	end

end

function BaseOptimizer:reset()

	self.optimizerInternalParameters = nil

end

return BaseOptimizer