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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseInstance = require("Core_BaseInstance")

local ModelParametersMerger = {}

ModelParametersMerger.__index = ModelParametersMerger

setmetatable(ModelParametersMerger, BaseInstance)

local defaultSplitMode = "Equal"

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

	NewModelParametersMerger.featureMatrixArray = parameterDictionary.featureMatrixArray or {}

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
		
		if (not featureMatrix) then error("No feature matrix.") end
		
		if (not labelVector) then error("No label vector.") end
		
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

local function mergeModelParametersNestedDictionaries(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension, ...)
	
	local tensorArray = {...}

	local dimensionSize = dimensionSizeArray[currentDimension] or 0
	
	local resultTable = {}

	if (currentDimension < numberOfDimensions) then
		
		-- Handle variable sizes by finding maximum index.
		
		local maximumIndex = 0
		
		for _, tensor in ipairs(tensorArray) do
			
			if (tensor) then maximumIndex = math.max(maximumIndex, #tensor) end
			
		end
		
		local actualSize = math.max(dimensionSize, maximumIndex)
		
		for i = 1, actualSize, 1 do
			
			local subTensorArray = {}
			
			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, (tensor and tensor[i]) or nil) end
			
			resultTable[i] = mergeModelParametersNestedDictionaries(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension + 1, table.unpack(subTensorArray))
			
		end
		
	elseif (currentDimension == numberOfDimensions) then
		
		local keyArray = {}
		
		for _, tensor in ipairs(tensorArray) do keyArray = collectKeys(tensor, keyArray) end
		
		for i, key in ipairs(keyArray) do
			
			local valueArray = {}
			
			for _, tensor in ipairs(tensorArray) do table.insert(valueArray, tensor[key] or 0) end
			
			resultTable[key] = functionToApply(table.unpack(valueArray))
			
		end
		
	else
		
		resultTable = functionToApply(table.unpack(tensorArray))
		
	end

	return resultTable
	
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
		
		if (keyCount == 1) then firstKey = i end
		
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

		NewModelParameters = mergeModelParametersNestedDictionaries(functionToApply, dimensionSizeArray, depth, 1, table.unpack(ModelParametersArray))

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

		NewModelParameters = mergeModelParametersNestedDictionaries(functionToApply, dimensionSizeArray, depth, 1, table.unpack(ModelParametersArray))

	elseif (isTableOfMatrices)  and (mergeMode == "Average") then

		NewModelParameters = calculateAverageModelParametersTable(ModelParametersArray)

	elseif (isMatrix) and (mergeMode == "Average") then

		NewModelParameters = createAverageModelParameters(ModelParametersArray)

	else

		error("Invalid merge mode.")

	end

	return NewModelParameters

end

local function mergeModelParametersUsingScalars(Model, modelType, mergeMode, splitMode, splitAmountArray, featureMatrix, labelVector, ModelParametersArray)
	
	if (not splitAmountArray) then splitAmountArray = getSplitAmountArray(Model, modelType, splitMode, featureMatrix, labelVector, ModelParametersArray) end

	return mergeModelParameters(mergeMode, ModelParametersArray, splitAmountArray)
	
end

local function getDotProductFeatureMatrixArray(featureMatrixArray)
	
	local dotProductFeatureMatrixArray = {}
	
	local transposedFeatureMatrix

	for i, featureMatrix in ipairs(featureMatrixArray) do
		
		transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
		
		dotProductFeatureMatrixArray[i] = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	end
	
	return dotProductFeatureMatrixArray
	
end

local function calculateRegressionMeanModelParameters(ModelParametersArray, dotProductFeatureMatrixArray)

	local sumDotProductFeatureMatrix

	local sumFeatureMatrixDotProductWeightMatrix

	local featureMatrixDotProductWeightMatrix

	for i, dotProductFeatureMatrix in ipairs(dotProductFeatureMatrixArray) do
		
		if (sumDotProductFeatureMatrix) then
			
			sumDotProductFeatureMatrix = AqwamTensorLibrary:add(sumDotProductFeatureMatrix, dotProductFeatureMatrix)
			
		else
			
			sumDotProductFeatureMatrix = dotProductFeatureMatrix
			
		end
		
		featureMatrixDotProductWeightMatrix = AqwamTensorLibrary:dotProduct(dotProductFeatureMatrix, ModelParametersArray[i])

		if (sumFeatureMatrixDotProductWeightMatrix) then
			
			sumFeatureMatrixDotProductWeightMatrix = AqwamTensorLibrary:add(sumFeatureMatrixDotProductWeightMatrix, featureMatrixDotProductWeightMatrix)
			
		else
			
			sumFeatureMatrixDotProductWeightMatrix = featureMatrixDotProductWeightMatrix
			
		end

	end

	local NewModelParametersPart1 = AqwamTensorLibrary:inverse(sumDotProductFeatureMatrix)

	local NewModelParameters = AqwamTensorLibrary:dotProduct(NewModelParametersPart1, sumFeatureMatrixDotProductWeightMatrix)

	return NewModelParameters

end

local function mergeModelParametersUsingRegressionMean(featureMatrixArray, ModelParametersArray)
	
	if (#featureMatrixArray ~= #ModelParametersArray) then error("The number of feature matrices does not equal to the number of model parameters.") end
	
	local dotProductFeatureMatrixArray = getDotProductFeatureMatrixArray(featureMatrixArray)
	
	local depth, hasDictionary = checkDepth(ModelParametersArray[1])
	
	if (hasDictionary) then error("The model parameters cannot have string keys.") end

	local isMatrix = (depth == 2)
	
	if (not isMatrix) then error("Can only perform regression mean on non-nested matrices.") end
	
	return calculateRegressionMeanModelParameters(ModelParametersArray, dotProductFeatureMatrixArray)
	
end

function ModelParametersMerger:merge(...)

	local ModelParametersArray = {...}

	if (#ModelParametersArray <= 0) then error("No model parameters set.") end
	
	local mergeMode = self.mergeMode
	
	local roundingMode = self.roundingMode
	
	local featureMatrixArray = self.featureMatrixArray
	
	local NewModelParameters
	
	if (mergeMode == "RegressionMean") then
		
		NewModelParameters = mergeModelParametersUsingRegressionMean(featureMatrixArray, ModelParametersArray)
		
	else
		
		NewModelParameters = mergeModelParametersUsingScalars(self.Model, self.modelType, mergeMode, self.splitMode,  self.splitAmountArray, featureMatrixArray[1], self.labelVector, ModelParametersArray)
		
	end
	
	if (roundingMode == "None") then return NewModelParameters end
	
	local roundingFunctionToApply = roundFunctionList[roundingMode]
	
	if (not roundingFunctionToApply) then error("Invalid rounding mode.") end
	
	local depth = checkDepth(NewModelParameters)

	return round(roundingFunctionToApply, NewModelParameters, depth, 1)

end

return ModelParametersMerger
