local StringSplitter = {}

local function addSpacesBetweenPattern(inputString, patternWhereToSplitBetween)
	
	local stringLength = string.len(inputString)
	
	local newString = ""
	
	local currentSubString
	
	local previousSubString
	
	local nextSubString
	
	local isCurrentSubStringMatched
	
	local isNextSubStringNotWhiteSpace
	
	local isPreviousSubStringNotWhiteSpace
	
	for index = 1, stringLength, 1 do
		
		currentSubString = string.sub(inputString, index, index)
		
		nextSubString = string.sub(inputString, index - 1, index - 1)

		previousSubString = string.sub(inputString, index + 1, index + 1)
		
		isCurrentSubStringMatched = string.find(currentSubString, patternWhereToSplitBetween)
		
		isNextSubStringNotWhiteSpace = not string.find(nextSubString, '%s')
			
		isPreviousSubStringNotWhiteSpace = not string.find(previousSubString, '%s')
		
		if isCurrentSubStringMatched then
			
			if isPreviousSubStringNotWhiteSpace then currentSubString = " " .. currentSubString end
			
			if isNextSubStringNotWhiteSpace then currentSubString = currentSubString .. " " end
			
		end
		
		newString = newString .. currentSubString
		
	end
	
	return newString
	
end

function StringSplitter:splitStringToArray(inputString, patternWhereToSplitBetween)
	
	if (typeof(inputString) ~= "string") then error("Input is not a string!") end
	
	local inputStringWithSpacesBetweenSymbols = addSpacesBetweenPattern(inputString, patternWhereToSplitBetween)
	
	local stringTable = string.split(inputString, " ")
	
	return stringTable
	
end

return StringSplitter

