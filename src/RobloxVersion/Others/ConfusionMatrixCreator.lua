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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

ConfusionMatrixCreator = {}

ConfusionMatrixCreator.__index = ConfusionMatrixCreator

setmetatable(ConfusionMatrixCreator, BaseInstance)

local calculateStatisticFunctionList = {
	
	["Precision"] = function(truePositiveCount, falsePositiveCount, falseNegativeCount, trueNegativeCount, beta) return ((truePositiveCount + falsePositiveCount == 0) and 0) or truePositiveCount / (truePositiveCount + falsePositiveCount) end,
	
	["Recall"] = function(truePositiveCount, falsePositiveCount, falseNegativeCount, trueNegativeCount, beta) return ((truePositiveCount + falseNegativeCount == 0) and 0) or truePositiveCount / (truePositiveCount + falseNegativeCount) end,	
	
	["Specificity"] = function(truePositiveCount, falsePositiveCount, falseNegativeCount, trueNegativeCount, beta) return ((trueNegativeCount + falsePositiveCount == 0) and 0) or trueNegativeCount / (trueNegativeCount + falsePositiveCount) end,
	
	["F"] = function(truePositiveCount, falsePositiveCount, falseNegativeCount, trueNegativeCount, beta)
		
		local squaredBeta = math.pow(beta, 2)
		
		local onePlusSquaredBeta = 1 + squaredBeta
		
		local numerator = onePlusSquaredBeta * truePositiveCount
		
		local denominator = (onePlusSquaredBeta * truePositiveCount) + (squaredBeta * falseNegativeCount) + falsePositiveCount
		
		return numerator / denominator
	end,
	
}

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

local function checkClassesList(ClassesList, trueLabelVector, predictedLabelVector)

	if (#ClassesList == 0) then

		ClassesList = createClassesList(trueLabelVector)

		local areNumbersOnly = areNumbersOnlyInList(ClassesList)

		if (areNumbersOnly) then table.sort(ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(trueLabelVector, ClassesList) then error("A value does not exist in the classes list is present in the true label vector.") end
		
		if checkIfAnyLabelVectorIsNotRecognized(predictedLabelVector, ClassesList) then error("A value does not exist in the classes list is present in the predicted label vector.") end
		
	end
	
	return ClassesList

end

local function generateTableText(indicatorString, RowsList, ColumnsList, matrix)
	
	local ExtendedColumnsList = table.clone(ColumnsList)
	
	table.insert(ExtendedColumnsList, 1, indicatorString)
	
	local maximumColumnValueLengthArray = {}

	for columnIndex, columnValue in ipairs(ExtendedColumnsList) do

		maximumColumnValueLengthArray[columnIndex] = string.len(tostring(columnValue))

	end
	
	for rowIndex, rowValue in ipairs(RowsList) do
		
		for columnIndex, columnValue in ipairs(ExtendedColumnsList) do
			
			maximumColumnValueLengthArray[columnIndex] = math.max(maximumColumnValueLengthArray[columnIndex], string.len(tostring(matrix[rowIndex][columnIndex]))) 
			
		end
		
	end

	local text =  "\n\n+"

	for columnIndex, columnValue in ipairs(ExtendedColumnsList) do

		local cellWidth = string.len(columnValue)

		local padding = maximumColumnValueLengthArray[columnIndex] + 2

		text = text .. string.rep("-", padding)

		text = text .. "+"

	end

	text = text .. "\n| "

	for columnIndex, columnValue in ipairs(ExtendedColumnsList) do

		local cellText = tostring(columnValue) 

		local cellWidth = string.len(cellText)

		local padding = maximumColumnValueLengthArray[columnIndex] - cellWidth

		text = text .. string.rep(" ", padding) .. cellText

		text = text .. " | "

	end

	text = text .. "\n+"

	for columnIndex, columnValue in ipairs(ExtendedColumnsList) do

		local cellWidth = string.len(columnValue)

		local padding = maximumColumnValueLengthArray[columnIndex] + 2

		text = text .. string.rep("-", padding)

		text = text .. "+"

	end

	text = text .. "\n" 
	
	for rowIndex, rowValue in ipairs(RowsList) do
		
		local cellRowHeaderText = tostring(rowValue) 

		local cellWidth = string.len(cellRowHeaderText)

		local columnRowPadding = maximumColumnValueLengthArray[1] - cellWidth + 1

		text = text .. "|" .. string.rep(" ", columnRowPadding) .. cellRowHeaderText .. " |"

		for columnIndex, value in ipairs(ColumnsList) do

			local cellValue = matrix[rowIndex][columnIndex]

			local cellText = tostring(cellValue) 

			local cellWidth = string.len(cellText)

			local padding = maximumColumnValueLengthArray[columnIndex + 1] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText

			text = text .. " |"

		end

		text = text .. "\n"
		
	end

	text = text .. "+"

	for columnIndex, columnValue in ipairs(ExtendedColumnsList) do

		local cellWidth = string.len(columnValue)

		local padding = maximumColumnValueLengthArray[columnIndex] + 2

		text = text .. string.rep("-", padding)

		text = text .. "+"

	end

	text = text .. "\n\n"

	return text
	
end

function ConfusionMatrixCreator.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewConfusionMatrixCreator = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewConfusionMatrixCreator, ConfusionMatrixCreator)
	
	NewConfusionMatrixCreator:setName("ConfusionMatrixCreator")
	
	NewConfusionMatrixCreator:setClassName("ConfusionMatrixCreator")
	
	NewConfusionMatrixCreator.ClassesList = parameterDictionary.ClassesList or {}
	
	return NewConfusionMatrixCreator
	
end

function ConfusionMatrixCreator:createConfusionMatrix(trueLabelVector, predictedLabelVector)
	
	if (#trueLabelVector ~= #predictedLabelVector) then error("The number of data are not equal!") end
	
	if (#trueLabelVector[1] ~= 1) or (#predictedLabelVector[1] ~= 1) then error("Both vector must only have one column!") end
	
	local ClassesList = checkClassesList(self.ClassesList, trueLabelVector, predictedLabelVector)
	
	self.ClassesList = ClassesList
	
	local numberOfClasses = #ClassesList
	
	local confusionMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfClasses}, 0)
	
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
	
	local text = generateTableText("True \\ Predicted", ClassesList, ClassesList, confusionMatrix)
	
	print(text)

	return confusionMatrix, numberOfUnknownClassifications
	
end

function ConfusionMatrixCreator:createCountMatrix(trueLabelVector, predictedLabelVector)
	
	local confusionMatrix = self:createConfusionMatrix(trueLabelVector, predictedLabelVector)

	local numberOfClasses = #self.ClassesList
	
	local totalNumberOfSamples = 0

	for i = 1, numberOfClasses do
		
		for j = 1, numberOfClasses do
			
			totalNumberOfSamples = totalNumberOfSamples + confusionMatrix[i][j]
			
		end
		
	end

	local countMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, 4}, 0)
	
	local truePositiveCount
	
	local falsePositiveCount
	
	local falseNegativeCount
	
	local trueNegativeCount
	
	for classIndex = 1, numberOfClasses do

		truePositiveCount = confusionMatrix[classIndex][classIndex]
		
		falsePositiveCount = 0
		
		falseNegativeCount = 0

		for i = 1, numberOfClasses do
			
			if (i ~= classIndex) then
				
				falsePositiveCount = falsePositiveCount + confusionMatrix[i][classIndex]
				
				falseNegativeCount = falseNegativeCount + confusionMatrix[classIndex][i]
				
			end
			
		end

		trueNegativeCount = totalNumberOfSamples - (truePositiveCount + falsePositiveCount + falseNegativeCount)

		countMatrix[classIndex][1] = truePositiveCount
		
		countMatrix[classIndex][2] = falsePositiveCount
		
		countMatrix[classIndex][3] = falseNegativeCount
		
		countMatrix[classIndex][4] = trueNegativeCount
		
	end

	return countMatrix, totalNumberOfSamples
	
end

function ConfusionMatrixCreator:printCountMatrix(trueLabelVector, predictedLabelVector)

	local countMatrix, totalNumberOfSamples = self:createCountMatrix(trueLabelVector, predictedLabelVector)
	
	local ClassesList = self.ClassesList
	
	local numberOfClasses = #ClassesList

	local HeaderStringList = {"True Positive", "False Positive", "False Negative", "True Negative"}
	
	local text = generateTableText("Class \\ Count Type", ClassesList, HeaderStringList, countMatrix)

	print(text)
	
	return countMatrix, totalNumberOfSamples
	
end

function ConfusionMatrixCreator:calculateStatistic(trueLabelVector, predictedLabelVector, statisticName)
	
	if (not statisticName) then error("Unknown statistic name.") end

	local numberOfClasses
	
	if (string.sub(statisticName, 1, 5) == "Macro") then
		
		local metricName = string.sub(statisticName, 6)
		
		local statisticValueVector = self:calculateStatistic(trueLabelVector, predictedLabelVector, metricName)

		local sum = 0
		
		numberOfClasses = #self.ClassesList
		
		for i = 1, numberOfClasses do sum = sum + statisticValueVector[i][1] end

		return sum / numberOfClasses

	end

	local countMatrix, totalNumberOfSamples = self:createCountMatrix(trueLabelVector, predictedLabelVector)
	
	numberOfClasses = #self.ClassesList
	
	if (statisticName == "Accuracy") then
		
		local correct = 0
		
		for i = 1, numberOfClasses do correct = correct + countMatrix[i][1] end
		
		return correct / totalNumberOfSamples
		
	end
	
	local statisticValueVector = AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)
	
	local beta
	
	if (string.sub(statisticName, 1, 1) == "F") then
		
		local betaString = string.sub(statisticName, 2)
		
		if (betaString == "") then
			
			beta = 1
			
		else
			
			beta = tonumber(betaString)
			
			if (not beta) then error("Invalid beta value in F-score name: " .. statisticName) end
			
		end
		
		statisticName = "F"
		
	end
	
	local calculateStatisticFunction = calculateStatisticFunctionList[statisticName]

	if (not calculateStatisticFunction) then error("Unknown statistic name.") end

	for i = 1, numberOfClasses do

		statisticValueVector[i][1] = calculateStatisticFunction(countMatrix[i][1], countMatrix[i][2], countMatrix[i][3], countMatrix[i][4], beta)

	end

	return statisticValueVector
	
end

function ConfusionMatrixCreator:printStatistics(trueLabelVector, predictedLabelVector, statisticNameArray)

	local statisticNameArray = statisticNameArray or {"Precision", "Recall", "Specificity", "F1"}
	
	for i, statisticName in ipairs(statisticNameArray) do
		
		if (string.sub(statisticName, 1, 5) == "Macro") then error("Cannot print macro statistics.") end
		
	end
	
	local statisticValueVectorArray = {}
	
	for i, statisticName in ipairs(statisticNameArray) do

		statisticValueVectorArray[i] = self:calculateStatistic(trueLabelVector, predictedLabelVector, statisticName)

	end
	
	local ClassesList = self.ClassesList
	
	local numberOfClasses = #ClassesList
	
	local numberOfStatistics = #statisticNameArray
	
	local statisticValueMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfStatistics})
	
	for i = 1, numberOfClasses, 1 do
		
		for j = 1, numberOfStatistics, 1 do
			
			statisticValueMatrix[i][j] = statisticValueVectorArray[j][i][1]
			
		end
		
	end
	
	local text = generateTableText("Class \\ Statistic", ClassesList, statisticNameArray, statisticValueMatrix)

	print(text)
	
	return statisticValueVectorArray
	
end

function ConfusionMatrixCreator:printMacroStatistics(trueLabelVector, predictedLabelVector, macroStatisticNameArray)
	
	local macroStatisticNameArray = macroStatisticNameArray or {"MacroPrecision", "MacroRecall", "MacroSpecificity", "MacroF1"}

	for i, macroStatisticName in ipairs(macroStatisticNameArray) do

		if (string.sub(macroStatisticName, 1, 5) ~= "Macro") then
			
			macroStatisticNameArray[i] = "Macro" .. macroStatisticName
			
		end

	end
	
	local unwrappedMacroStatisticMatrix = {}

	for i, statisticName in ipairs(macroStatisticNameArray) do

		unwrappedMacroStatisticMatrix[i] = self:calculateStatistic(trueLabelVector, predictedLabelVector, statisticName)

	end
	
	local text = generateTableText("Statistic", {"Value"}, macroStatisticNameArray, {unwrappedMacroStatisticMatrix})
	
	print(text)
	
	return unwrappedMacroStatisticMatrix
	
end

return ConfusionMatrixCreator
