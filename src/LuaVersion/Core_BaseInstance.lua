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

local AHAMachineLearningAndDeepLearningLibraryInstance = {}

AHAMachineLearningAndDeepLearningLibraryInstance.__index = AHAMachineLearningAndDeepLearningLibraryInstance

local currentID = 0

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

local function incrementIDs(object)
	
	if (type(object) == "table") then
		
		if (object.id) then
			
			currentID = currentID + 1
			
			object.id = currentID
			
		end
		
		for _, value in pairs(object) do incrementIDs(value) end
		
	end
	
end

function AHAMachineLearningAndDeepLearningLibraryInstance.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	if (type(parameterDictionary) ~= "table") then error("The constructor requires a parameter dictionary (table).") end
	
	local NewAHAMachineLearningAndDeepLearningLibraryInstance = {}

	setmetatable(NewAHAMachineLearningAndDeepLearningLibraryInstance, AHAMachineLearningAndDeepLearningLibraryInstance)
	
	currentID = currentID + 1
	
	NewAHAMachineLearningAndDeepLearningLibraryInstance.id = currentID
	
	NewAHAMachineLearningAndDeepLearningLibraryInstance.name = "Unknown"

	NewAHAMachineLearningAndDeepLearningLibraryInstance.className = "Unknown"
	
	return NewAHAMachineLearningAndDeepLearningLibraryInstance
	
end

function AHAMachineLearningAndDeepLearningLibraryInstance:getID()

	return self.id

end

function AHAMachineLearningAndDeepLearningLibraryInstance:setName(name)

	self.name = name

end

function AHAMachineLearningAndDeepLearningLibraryInstance:getName()

	return self.name

end

function AHAMachineLearningAndDeepLearningLibraryInstance:setClassName(className)

	self.className = className

end

function AHAMachineLearningAndDeepLearningLibraryInstance:getClassName()

	return self.className

end

function AHAMachineLearningAndDeepLearningLibraryInstance:getValueOrDefaultValue(value, defaultValue)

	if (type(value) == "nil") then return defaultValue end

	return value

end

function AHAMachineLearningAndDeepLearningLibraryInstance:deepCopyTable(original)

	return deepCopyTable(original)

end

function AHAMachineLearningAndDeepLearningLibraryInstance:clone()
	
	local clonedInstance = deepCopyTable(self)
	
	incrementIDs(clonedInstance)
	
	return clonedInstance
	
end

function AHAMachineLearningAndDeepLearningLibraryInstance:destroy()
	
	setmetatable(self, nil)

	table.clear(self)

	self = nil
	
end

return AHAMachineLearningAndDeepLearningLibraryInstance
