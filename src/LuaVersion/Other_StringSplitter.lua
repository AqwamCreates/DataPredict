--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
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
