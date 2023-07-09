local StringSplitter = {}

local function addSpacesBetweenSymbols(inputString)
	
	local index
	
	local symbolIndex
	
	local symbol
	
	local previousCharacter
	
	repeat
		
		index = string.match(inputString, '[%a%d%p]%p')
		
		if index then
			
			symbolIndex = symbolIndex + 1
			
			symbol = inputString[symbolIndex]
			
			previousCharacter = inputString[previousCharacter]
			
			string.gsub(inputString, '[%a%d%p]' .. symbol, previousCharacter .. ' ' .. symbol)
			
		end

	until (index == nil)
	
	return inputString
	
end

local function convertStringToTable(inputString)
	
	local stringTable = {}
	
	for key, value in string.gmatch(inputString, "(%w+)=(%w+)") do
		
		table.insert(stringTable, value)
		
	end
	
	return stringTable
	
end

function StringSplitter:splitToArray(inputString)
	
	local inputStringWithSpacesBetweenSymbols = addSpacesBetweenSymbols(inputString)
	
	local stringTable = convertStringToTable(inputStringWithSpacesBetweenSymbols)
	
	return stringTable
	
end

return StringSplitter
