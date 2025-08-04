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
	
	local newInputString = addSpacesBetweenPattern(inputString, patternWhereToSplitBetween)
	
	local stringTable = string.split(newInputString, " ")
	
	if (stringTable[1] == "") then table.remove(stringTable, 1) end
	
	return stringTable
	
end

return StringSplitter
