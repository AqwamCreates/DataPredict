function zTableFunction:getStandardNormalInverseCumulativeDistributionFunction(probability)
	
	local rowStringFormat = rowStringFormat

	local columnStringFormat = columnStringFormat
	
	local stringFormatFunction = string.format
	
	local isFlipped = (probability > 0.5)

	local finalProbability = (isFlipped and (1 - probability) or probability)
	
	local clampedProbability = math.clamp(finalProbability, 0.00005, 0.5)

	local closestZValue
	
	local rowString
	
	local rowTable
	
	local columnValueArray
	
	local columnValue1
	
	local columnValue2
	
	local probabilityValue1
	
	local probabilityValue2
	
	local fraction
	
	local closestZValue

	for _, rowValue in ipairs(cachedRowValueArray) do
		
		rowString = stringFormatFunction(rowStringFormat, rowValue)
		
		rowTable = zTable[rowString]

		columnValueArray = {}
		
		for k in pairs(rowTable) do table.insert(columnValueArray, tonumber(k)) end
		
		table.sort(columnValueArray)

		for i = 1, #columnValueArray - 1 do
			
			columnValue1 = columnValueArray[i]
			
			columnValue2 = columnValueArray[i + 1]
			
			probabilityValue1 = rowTable[stringFormatFunction(columnStringFormat, columnValue1)]
			
			probabilityValue2 = rowTable[stringFormatFunction(columnStringFormat, columnValue2)]

			if (clampedProbability >= probabilityValue1) and (clampedProbability <= probabilityValue2) then

				fraction = (clampedProbability - probabilityValue1) / (probabilityValue2 - probabilityValue1)
				
				closestZValue = rowValue - (columnValue1 + (fraction * (columnValue2 - columnValue1)))
				
				break
				
			end
			
		end
		
		if (closestZValue) then break end
		
	end
	
	if (isFlipped) and (closestZValue) then closestZValue = -closestZValue end

	return closestZValue
	
end
