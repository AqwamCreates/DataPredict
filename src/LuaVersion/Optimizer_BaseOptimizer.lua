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
	
	NewBaseOptimizer.calculateFunction = nil
	
	NewBaseOptimizer.optimizerInternalParameters = {}
	
	return NewBaseOptimizer
	
end

function BaseOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	if (self.calculateFunction) then return self.calculateFunction(learningRate, costFunctionDerivatives) end
	
end

function BaseOptimizer:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
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

function BaseOptimizer:setOptimizerInternalParameters(optimizerInternalParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.optimizerInternalParameters = optimizerInternalParameters

	else

		self.optimizerInternalParameters = deepCopyTable(optimizerInternalParameters)

	end

end

function BaseOptimizer:reset()

	self.optimizerInternalParameters = {}

end

return BaseOptimizer
