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

BaseValueScheduler = {}

BaseValueScheduler.__index = BaseValueScheduler

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


function BaseValueScheduler.new(valueSchedulerName)
	
	local NewBaseValueScheduler = {}
	
	setmetatable(NewBaseValueScheduler, BaseValueScheduler)
	
	NewBaseValueScheduler.valueSchedulerName = valueSchedulerName or "Unknown"
	
	NewBaseValueScheduler.calculateFunction = nil
	
	NewBaseValueScheduler.epsilonModifierInternalParameters = nil
	
	return NewBaseValueScheduler
	
end

function BaseValueScheduler:calculate(epsilon)
	
	if (self.calculateFunction) then return self.calculateFunction(epsilon) end
	
end

function BaseValueScheduler:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseValueScheduler:getValueSchedulerName()
	
	return self.valueSchedulerName
	
end

function BaseValueScheduler:getValueSchedulerInternalParameters(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.valueSchedulerInternalParameters
		
	else
		
		return deepCopyTable(self.valueSchedulerInternalParameters)
		
	end
	
end

function BaseValueScheduler:setValueSchedulerInternalParameters(valueSchedulerInternalParameters, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.valueSchedulerInternalParameters = valueSchedulerInternalParameters

	else

		self.valueSchedulerInternalParameters = deepCopyTable(valueSchedulerInternalParameters)

	end

end

function BaseValueScheduler:reset()

	self.valueSchedulerInternalParameters = nil

end

return BaseValueScheduler
