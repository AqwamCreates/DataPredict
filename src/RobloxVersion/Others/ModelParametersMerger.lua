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

local ModelParametersMerger = {}

ModelParametersMerger.__index = ModelParametersMerger

setmetatable(ModelParametersMerger, BaseInstance)

local defaultSplitMode = "Accuracy"

local defaultMergeMode = "Average"

local defaultRoundingMode = "None"

local roundFunctionList = {
	
	["Floor"] = math.floor,
	
	["Ceiling"] = math.ceil,
	
}

function ModelParametersMerger.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewModelParametersMerger = BaseInstance.new(parameterDictionary)

	setmetatable(NewModelParametersMerger, ModelParametersMerger)

	NewModelParametersMerger:setName("ModelParametersMerger")

	NewModelParametersMerger:setClassName("ModelParametersMerger")

	NewModelParametersMerger.Model = parameterDictionary.Model

	NewModelParametersMerger.modelType = parameterDictionary.modelType

	ModelParametersMerger.splitMode = parameterDictionary.splitMode or defaultSplitMode

	NewModelParametersMerger.mergeMode = parameterDictionary.mergeMode or defaultMergeMode

	NewModelParametersMerger.roundingMode = parameterDictionary.roundingMode or defaultRoundingMode

	NewModelParametersMerger.featureMatrix = parameterDictionary.featureMatrix

	NewModelParametersMerger.labelVector = parameterDictionary.labelVector

	NewModelParametersMerger.splitAmountArray = parameterDictionary.splitAmountArray

	return NewModelParametersMerger

end

function ModelParametersMerger:setCustomSplitAmountArray(splitAmountArray)

	self.splitAmountArray = splitAmountArray or self.splitAmountArray

end

function ModelParametersMerger:setData(featureMatrix, labelVector)

	if (featureMatrix) and (labelVector) then

		if (#featureMatrix ~= #labelVector) then error("Feature matrix and the label vector does not contain the same number of rows.") end

	end

	self.featureMatrix = featureMatrix or self.featureMatrix

	self.labelVector = labelVector or self.labelVector

end

local function round(functionToApply, nestedTable, numberOfDimensions, currentDimension) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local resultTensor = {}

	if (currentDimension < numberOfDimensions) then

		for i, subTensor in pairs(nestedTable) do resultTensor[i] = round(functionToApply, subTensor, numberOfDimensions, currentDimension + 1) end

	elseif (currentDimension == numberOfDimensions) then -- Much more efficient than applying recursion again to get the original value.

		for i, value in pairs(nestedTable) do resultTensor[i] = functionToApply(value) end

	else -- Sometimes the original tensor can be a number, so we must do the operation directly.

		resultTensor = functionToApply(nestedTable)

	end

	return resultTensor

end

local function checkDepth(array, depth, hasDictionary)

	depth = depth or 0

	hasDictionary = hasDictionary or false

	local valueType = typeof(array)

	if (valueType == "table") then

		-- Check if this table is a dictionary (has non-integer keys or isn't a simple sequence).

		local isDictionary = false

		local numericCount = 0

		for key, _ in pairs(array) do

			if (type(key) == "number") and (key >= 1 ) and (math.floor(key) == key) then

				numericCount = numericCount + 1

			else

				isDictionary = true

				break

			end

		end

		-- Check if it's a sequence (consecutive integer keys starting from 1).
		
		if (not isDictionary) and (numericCount > 0) then

			local isSequence = true

			for i = 1, numericCount, 1 do

				if (not array[i]) then

					isSequence = false

					break

				end

			end

			isDictionary = (not isSequence)

		end

		-- Recurse into the first element if it's an array/sequence.

		if (not isDictionary) and (array[1]) then

			return checkDepth(array[1], depth + 1, hasDictionary)

		else

			-- This is a dictionary, so the next level would require a key.

			return (depth + 1), true

		end

	else

		-- Reached a non-table value.

		return depth, hasDictionary

	end

end

local function generateModelParametersTableWithMatricesOfZeroValues(ModelParameters)

	local NewModelParameters = {}

	for i, matrix in ipairs(ModelParameters) do

		local numberOfRows = #matrix

		local numberOfColumns = #matrix[1]

		local newMatrix = AqwamTensorLibrary:createTensor({numberOfRows, numberOfColumns})

		table.insert(NewModelParameters, newMatrix)

	end

	return NewModelParameters

end

local function calculateTotalFromArray(array)

	local total = 0

	for i, value in ipairs(array) do total = total + value end

	return total

end

local function convertValueArrayToPercentageArray(array)

	local percentage

	local total = calculateTotalFromArray(array)

	local percentageArray = {}

	for i, value in ipairs(array) do

		if (total == 0) then

			percentage = 0

		else

			percentage = math.abs(value / total)

		end

		table.insert(percentageArray, percentage)

	end

	return percentageArray

end

local function generateErrorArrayForRegression(Model, ModelParametersArray, featureMatrix, labelVector)

	local errorArray = {}

	for i, ModelParameters in ipairs(ModelParametersArray) do

		Model:setModelParameters(ModelParameters)

		local predictVector = Model:predict(featureMatrix)

		local errorVector = AqwamTensorLibrary:subtract(labelVector, predictVector)

		local absoluteErrorVector = AqwamTensorLibrary:applyFunction(math.abs, errorVector)

		local errorValue = AqwamTensorLibrary:sum(absoluteErrorVector)

		table.insert(errorArray, errorValue)

	end

	return errorArray

end

local function generateErrorArrayForClustering(Model, ModelParametersArray, featureMatrix)

	local errorArray = {}

	for i, ModelParameters in ipairs(ModelParametersArray) do

		Model:setModelParameters(ModelParameters)

		local _, distanceVector = Model:predict(featureMatrix)

		local errorValue = AqwamTensorLibrary:sum(distanceVector)

		table.insert(errorArray, errorValue)

	end

	return errorArray

end

local function convertErrorArrayToAccuracyArray(errorArray)

	local accuracyArray = {}

	local errorPercentageArray = convertValueArrayToPercentageArray(errorArray)

	for i, errorPercentage in ipairs(errorPercentageArray) do

		local accuracy = 1 - errorPercentage

		table.insert(accuracyArray, accuracy)

	end

	return accuracyArray

end

local function generateAccuracyArrayForClassification(Model, ModelParametersArray, featureMatrix, labelVector)

	local accuracyArray = {}

	local totalLabel = #labelVector

	for i, ModelParameters in ipairs(ModelParametersArray) do

		local accuracy = 0

		local totalCorrect = 0

		Model:setModelParameters(ModelParameters)

		local predictedlabelVector = Model:predict(featureMatrix)

		for j = 1, totalLabel, 1 do

			if (predictedlabelVector[j][1] == labelVector[j][1]) then totalCorrect += 1 end

		end

		accuracy = totalCorrect / totalLabel

		table.insert(accuracyArray, accuracy)

	end

	return accuracyArray

end

local function checkIfAllValuesAreZeroesInArray(array)

	local allZeroes = true

	for i, accuracyPercentage in ipairs(array) do

		array = (accuracyPercentage == 0)

		if (not allZeroes) then break end

	end

	return allZeroes

end

local function generateAccuracyArray(Model, modelType, ModelParametersArray, featureMatrix, labelVector)

	if (not Model) then error("No model.") end

	if (not modelType) then error("No model type.") end

	local accuracyArray

	if (modelType == "Regression") then

		local errorArray = generateErrorArrayForRegression(Model, ModelParametersArray, featureMatrix, labelVector)

		accuracyArray = convertErrorArrayToAccuracyArray(errorArray)

	elseif (modelType == "Classification") then

		accuracyArray = generateAccuracyArrayForClassification(Model, ModelParametersArray, featureMatrix, labelVector)

	elseif (modelType == "Clustering") then

		local errorArray = generateErrorArrayForClustering(Model, ModelParametersArray, featureMatrix)

		accuracyArray = convertErrorArrayToAccuracyArray(errorArray)

	else

		error("Invalid model type!")

	end

	return accuracyArray

end

local function getIndexOfHighestAccuracy(accuracyArray)

	local index

	local highestAccuracy = -math.huge

	for i, accuracy in ipairs(accuracyArray)  do

		if (accuracy > highestAccuracy) then 

			highestAccuracy = accuracy 

			index = i

		end

	end

	return index

end

local function getSplitAmountArray(Model, modelType, splitMode, featureMatrix, labelVector, ModelParametersArray)
	
	local numberOfModelParameters = #ModelParametersArray
	
	local accuracyArray

	local splitAmountArray

	if (splitMode ~= "Equal") and (splitMode ~= "Ignore") then
		
		accuracyArray = generateAccuracyArray(Model, modelType, ModelParametersArray, featureMatrix, labelVector) 
		
	end

	if (splitMode == "Best") then

		local areAllZeroes = checkIfAllValuesAreZeroesInArray(accuracyArray)

		local bestModelParametersIndex

		if (areAllZeroes) then 

			bestModelParametersIndex = Random.new():NextInteger(1, numberOfModelParameters)

		else

			bestModelParametersIndex = getIndexOfHighestAccuracy(accuracyArray)

		end

		splitAmountArray = table.create(numberOfModelParameters, 0)

		splitAmountArray[bestModelParametersIndex] = 1

	elseif (splitMode == "Ratio") then

		splitAmountArray = convertValueArrayToPercentageArray(accuracyArray)

	elseif (splitMode == "Equal") then

		local average = 1 / numberOfModelParameters

		splitAmountArray = table.create(numberOfModelParameters, average)

	elseif (splitMode == "Ignore") then

		splitAmountArray = {}

	else

		error("Invalid split mode.")

	end

	return splitAmountArray

end

local function applyFunctionToEachMatricesInModelParameters(functionToApply, ModelParameters)

	for k, matrix in ipairs(ModelParameters) do

		ModelParameters[k] =  AqwamTensorLibrary:applyFunction(functionToApply, matrix)

	end

	return ModelParameters

end

local function collectKeys(dictionary, keyArray)
	
	for key, _ in pairs(dictionary) do
		
		if (not table.find(keyArray, key)) then
			
			table.insert(keyArray, key)
			
		end
		
	end
	
	return keyArray
	
end

local function mergeModelParametersNestedDictionary(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension, ...)
	
	local tensorArray = {...}

	-- Handle the case where we might be dealing with dictionaries.

	-- Check if we're dealing with dictionaries at this level.
	
	local firstTensor = tensorArray[1]

	-- Original array processing with variable dimension handling.
	
	local dimensionSize = dimensionSizeArray[currentDimension] or 0
	
	local resultArray = {}

	if (currentDimension < numberOfDimensions) then
		
		-- Handle variable sizes by finding max index.
		
		local maximumIndex = 0
		
		for _, tensor in ipairs(tensorArray) do
			
			if (tensor) then maximumIndex = math.max(maximumIndex, #tensor) end
			
		end

		-- Use the larger of dimensionSize or maximumIndex found.
		
		local actualSize = math.max(dimensionSize, maximumIndex)
		
		
		for i = 1, actualSize do
			
			local subTensorArray = {}
			
			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, (tensor and tensor[i]) or nil) end
			
			resultArray[i] = mergeModelParametersNestedDictionary(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension + 1, table.unpack(subTensorArray))
			
		end
		
	elseif (currentDimension == numberOfDimensions) then
		
		-- Last dimension - apply function directly.
		
		local keyArray = {}
		
		for _, tensor in ipairs(tensorArray) do keyArray = collectKeys(tensor, keyArray) end
		
		for i, key in ipairs(keyArray) do
			
			local valueArray = {}
			
			for _, tensor in ipairs(tensorArray) do table.insert(valueArray, tensor[key] or 0) end
			
			resultArray[key] = functionToApply(table.unpack(valueArray))
			
		end
		
	else
		
		resultArray = functionToApply(table.unpack(tensorArray))
		
	end

	return resultArray
	
end

local function calculateWeightedAverageModelParametersTable(ModelParametersArray, splitAmountArray)

	local totalSplitAmount = calculateTotalFromArray(splitAmountArray)

	local NewModelParametersTable = generateModelParametersTableWithMatricesOfZeroValues(ModelParametersArray[1])

	for i, ModelParametersTable in ipairs(ModelParametersArray) do

		for j, matrix in ipairs(ModelParametersTable) do

			local calculatedMatrix = AqwamTensorLibrary:multiply(splitAmountArray[i], matrix)

			NewModelParametersTable[j] = AqwamTensorLibrary:add(NewModelParametersTable[j], calculatedMatrix)

		end

	end

	for i, matrix in ipairs(NewModelParametersTable) do

		NewModelParametersTable[i] = AqwamTensorLibrary:divide(NewModelParametersTable[i], totalSplitAmount)

	end

	return NewModelParametersTable

end

local function calculateWeightedAverageModelParameters(ModelParametersArray, splitAmountArray)

	local totalSplitAmount = calculateTotalFromArray(splitAmountArray)

	local FirstModelParameters = ModelParametersArray[1]

	local NewModelParameters = AqwamTensorLibrary:createTensor({#FirstModelParameters, #FirstModelParameters[1]})

	for j, splitAmount in ipairs(splitAmountArray) do

		local matrix = ModelParametersArray[j]

		local calculatedMatrix = AqwamTensorLibrary:multiply(splitAmount, matrix)

		NewModelParameters = AqwamTensorLibrary:add(NewModelParameters, calculatedMatrix)

	end

	NewModelParameters = AqwamTensorLibrary:divide(NewModelParameters, totalSplitAmount)

	return NewModelParameters

end

local function calculateAverageModelParametersTable(ModelParametersArray)

	local NewModelParametersTable = generateModelParametersTableWithMatricesOfZeroValues(ModelParametersArray[1])

	local numberOfModelParameters = #ModelParametersArray

	for i, ModelParametersTable in ipairs(ModelParametersArray) do

		for j, matrix in ipairs(ModelParametersTable) do

			NewModelParametersTable[j] = AqwamTensorLibrary:add(NewModelParametersTable[j], matrix)

		end

	end

	for i, matrix in ipairs(NewModelParametersTable) do

		NewModelParametersTable[i] = AqwamTensorLibrary:divide(NewModelParametersTable[i], numberOfModelParameters)

	end

	return NewModelParametersTable

end

local function createAverageModelParameters(ModelParametersArray)

	local NewModelParameters = AqwamTensorLibrary:add(table.unpack(ModelParametersArray))

	local numberOfModelParameters = #ModelParametersArray

	NewModelParameters = AqwamTensorLibrary:divide(NewModelParameters, numberOfModelParameters)

	return NewModelParameters

end

local function getRecursiveDimensionSizeArray(tensor, targetDimensionSizeArray)

	if (type(tensor) ~= "table") then return end
	
	local keyCount = 0
	
	local firstKey
	
	for i, tensor in pairs(tensor) do
		
		keyCount = keyCount + 1
		
		if keyCount == 1 then firstKey = i	end
		
	end

	table.insert(targetDimensionSizeArray, keyCount)

	getRecursiveDimensionSizeArray(tensor[firstKey], targetDimensionSizeArray)

end

local function getDimensionSizeArray(tensor)

	local dimensionSizeArray = {}

	getRecursiveDimensionSizeArray(tensor, dimensionSizeArray)

	return dimensionSizeArray

end

local function mergeModelParameters(mergeMode, ModelParametersArray, splitAmountArray)

	local NewModelParameters

	local numberOfModelParameters = #ModelParametersArray

	local FirstModelParameters = ModelParametersArray[1]

	local depth, hasDictionary = checkDepth(FirstModelParameters)
	
	local dimensionSizeArray = getDimensionSizeArray(FirstModelParameters)

	local isTableOfMatrices = (depth == 3)

	local isMatrix = (depth == 2)

	if (hasDictionary) and (mergeMode == "WeightedAverage") then 
		
		local totalSplitAmount = calculateTotalFromArray(splitAmountArray)

		local functionToApply = function (...)

			local sum = 0

			for i, value in ipairs({...}) do sum = sum + (splitAmountArray[i] * value) end

			local weightedAverage = sum / totalSplitAmount

			return weightedAverage

		end

		NewModelParameters = mergeModelParametersNestedDictionary(functionToApply, dimensionSizeArray, depth, 1, table.unpack(ModelParametersArray))

	elseif (isTableOfMatrices) and (mergeMode == "WeightedAverage") then

		NewModelParameters = calculateWeightedAverageModelParametersTable(ModelParametersArray, splitAmountArray)

	elseif (isMatrix) and (mergeMode == "WeightedAverage") then

		NewModelParameters = calculateWeightedAverageModelParameters(ModelParametersArray, splitAmountArray)

	elseif (hasDictionary) and (mergeMode == "Average") then
		
		local functionToApply = function (...)
			
			local sum = 0
			
			for _, value in ipairs({...}) do sum = sum + value end
			
			local average = sum / numberOfModelParameters
			
			return average
			
		end

		NewModelParameters = mergeModelParametersNestedDictionary(functionToApply, dimensionSizeArray, depth, 1, table.unpack(ModelParametersArray))

	elseif (isTableOfMatrices)  and (mergeMode == "Average") then

		NewModelParameters = calculateAverageModelParametersTable(ModelParametersArray)

	elseif (isMatrix) and (mergeMode == "Average") then

		NewModelParameters = createAverageModelParameters(ModelParametersArray)

	else

		error("Invalid merge mode.")

	end

	return NewModelParameters

end

function ModelParametersMerger:merge(...)

	local ModelParametersArray = {...}

	if (#ModelParametersArray <= 0) then error("No model parameters set.") end

	local splitAmountArray = self.splitAmountArray

	if (not splitAmountArray) then

		splitAmountArray = getSplitAmountArray(self.Model, self.modelType, self.splitMode, self.featureMatrix, self.labelVector, ModelParametersArray)

	end

	local NewModelParameters = mergeModelParameters(self.mergeMode, ModelParametersArray, splitAmountArray)
	
	local roundingMode = self.roundingMode
	
	if (roundingMode == "None") then return NewModelParameters end
	
	local roundingFunctionToApply = roundFunctionList[roundingMode]
	
	if (not roundingFunctionToApply) then error("Invalid rounding mode.") end

	return round(roundingFunctionToApply, NewModelParameters, 1, 1)

end

return ModelParametersMerger
