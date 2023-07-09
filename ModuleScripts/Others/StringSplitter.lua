local StringSplitter = {}

local function addSpacesBetweenSymbols(inputString)
	
	local stringLength = string.len(inputString)
	
	local newString = ""
	
	local currentSubString
	
	local previousSubString
	
	local nextSubString
	
	local isCurrentSubStringASymbol
	
	local isNextSubStringNotSymbol
	
	local isPreviousSubStringNotSymbol
	
	for index = 1, stringLength, 1 do
		
		currentSubString = string.sub(inputString, index, index)
		
		nextSubString = string.sub(inputString, index - 1, index - 1)

		previousSubString = string.sub(inputString, index + 1, index + 1)
		
		isCurrentSubStringASymbol = string.find(currentSubString, '%p')
		
		isNextSubStringNotSymbol =  string.find(nextSubString, '[%a%d]')
			
		isPreviousSubStringNotSymbol = string.find(previousSubString, '[%a%d]')
		
		if isCurrentSubStringASymbol then
			
			if isPreviousSubStringNotSymbol then currentSubString = " " .. currentSubString end
			
			if isNextSubStringNotSymbol then currentSubString = currentSubString .. " " end
			
		end
		
		newString = newString .. currentSubString
		
	end
	
	return newString
	
end

local function convertStringToTable(inputString)
	
	local stringTable = {}
	
	local stringTableWithSpaces = string.split(inputString, " ")
	
	for i, value in ipairs(stringTableWithSpaces)  do
		
		if (value ~= "") then table.insert(stringTable, value) end
		
	end
	
	return stringTable
	
end

function StringSplitter:splitStringToArray(inputString)
	
	if (typeof(inputString) ~= "string") then error("Input is not a string!") end
	
	local inputStringWithSpacesBetweenSymbols = addSpacesBetweenSymbols(inputString)
	
	local stringTable = convertStringToTable(inputStringWithSpacesBetweenSymbols)
	
	return stringTable
	
end

return StringSplitter
