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

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ConfusionMatrixCreator = {}

ConfusionMatrixCreator.__index = ConfusionMatrixCreator

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

local function createClassesList(labelVector)

	local ClassesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(ClassesList, value) then

			table.insert(ClassesList, value)

		end

	end

	return ClassesList

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if table.find(ClassesList, labelVector[i][1]) then continue end

		return true

	end

	return false

end

function ConfusionMatrixCreator:checkLabelVectors(trueLabelVector, predictedLabelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(trueLabelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(trueLabelVector, self.ClassesList) then error("A value does not exist in the classes list is present in the true label vector.") end
		
		if checkIfAnyLabelVectorIsNotRecognized(predictedLabelVector, self.ClassesList) then error("A value does not exist in the classes list is present in the predicted label vector.") end
		
	end

end

function ConfusionMatrixCreator.new(ClassesList)
	
	local NewConfusionMatrixCreator = {}
	
	setmetatable(NewConfusionMatrixCreator, ConfusionMatrixCreator)
	
	NewConfusionMatrixCreator.ClassesList = ClassesList or {}
	
	return NewConfusionMatrixCreator
	
end

function ConfusionMatrixCreator:setParameters(ClassesList)
	
	self.ClassesList = ClassesList or self.ClassesList
	
end

function ConfusionMatrixCreator:createConfusionMatrix(trueLabelVector, predictedLabelVector)
	
	if (#trueLabelVector ~= #predictedLabelVector) then error("The number of data are not equal!") end
	
	if (#trueLabelVector[1] ~= 1) or (#predictedLabelVector[1] ~= 1) then error("Both vector must only have one column!") end
	
	self:checkLabelVectors(trueLabelVector, predictedLabelVector)
	
	local ClassesList = self.ClassesList
	
	local confusionMatrix = AqwamMatrixLibrary:createMatrix(#ClassesList, #ClassesList)
	
	local numberOfUnknownClassifications = 0
	
	for i = 1, #trueLabelVector, 1 do -- row: true value, column: predictedLabel
		
		local trueLabel = trueLabelVector[i][1]
			
		local predictedLabel = predictedLabelVector[i][1]
			
		local trueClassIndex = table.find(ClassesList, trueLabel)
			
		local predictedClassIndex = table.find(ClassesList, predictedLabel)
		
		if (trueClassIndex) and (predictedClassIndex) then
			
			confusionMatrix[trueClassIndex][predictedClassIndex] = confusionMatrix[trueClassIndex][predictedClassIndex] + 1
			
		else
			
			numberOfUnknownClassifications = numberOfUnknownClassifications + 1
			
		end
			
	end
	
	return confusionMatrix, numberOfUnknownClassifications
	
end

function ConfusionMatrixCreator:printConfusionMatrix(trueLabelVector, predictedLabelVector)
	
	local confusionMatrix, numberOfUnknownClassifications = self:createConfusionMatrix(trueLabelVector, predictedLabelVector)
	
	local ClassesList = self.ClassesList
	
	local numberOfClasses = #ClassesList
	
	local maxClassLabelLengthArray = {}
	
	local maxColumnValueLength = 3

	for i, classLabel in ipairs(ClassesList) do
		
		local length = string.len(tostring(classLabel))
		
		maxClassLabelLengthArray[i] = length
		
		maxColumnValueLength = math.max(maxColumnValueLength, length)
		
	end

	for column = 1, #confusionMatrix[1], 1 do
		
		for row = 1, #confusionMatrix, 1 do
			
			maxClassLabelLengthArray[column] = math.max(maxClassLabelLengthArray[column], string.len(tostring(confusionMatrix[row][column]))) 

		end
		
	end
	
	local text =  "\n\n" .. string.rep(" ", maxColumnValueLength + 3) .. "+"
	
	for i, classLabel in ipairs(ClassesList) do

		local cellWidth = string.len(classLabel)

		local padding = maxClassLabelLengthArray[i] + 2

		text = text .. string.rep("-", padding)

		text = text .. "+"

	end
	
	text = text .. "\n " .. tostring(" ", maxColumnValueLength - 1) .. "T\\P" .. " |"
	
	for i, classLabel in ipairs(ClassesList) do
		
		local cellText = tostring(classLabel) 
		
		local cellWidth = string.len(classLabel)
		
		local padding = maxClassLabelLengthArray[i] - cellWidth + 1
		
		text = text .. string.rep(" ", padding) .. cellText

		text = text .. " |"

	end
	
	text = text .. "\n+".. string.rep("-", maxColumnValueLength  + 2) .. "+"
	
	for i, classLabel in ipairs(ClassesList) do

		local cellWidth = string.len(classLabel)

		local padding = maxClassLabelLengthArray[i] + 2

		text = text .. string.rep("-", padding)

		text = text .. "+"

	end
	
	text = text .. "\n" 

	for row = 1, numberOfClasses, 1 do
		
		local cellRowHeaderText = tostring(ClassesList[row]) 

		local cellWidth = string.len(cellRowHeaderText)

		local columnRowPadding = maxColumnValueLength - cellWidth + 1

		text = text .. "|" .. string.rep(" ", columnRowPadding) .. cellRowHeaderText .. " |"

		for column = 1, numberOfClasses, 1 do

			local cellValue = confusionMatrix[row][column]

			local cellText = tostring(cellValue) 

			local cellWidth = string.len(cellText)

			local padding = maxClassLabelLengthArray[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText
			
			text = text .. " |"

		end

		text = text .. "\n"

	end
	
	text = text .. "+" .. string.rep("-", maxColumnValueLength + 2) .. "+"
	
	for i, classLabel in ipairs(ClassesList) do

		local cellWidth = string.len(classLabel)

		local padding = maxClassLabelLengthArray[i] + 2

		text = text .. string.rep("-", padding)

		text = text .. "+"

	end
	
	text = text .. "\n\n"
	
	print(text)
	
	return confusionMatrix, numberOfUnknownClassifications
	
end

return ConfusionMatrixCreator