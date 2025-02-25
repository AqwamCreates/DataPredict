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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local ModelParametersMerger = {}

ModelParametersMerger.__index = ModelParametersMerger

setmetatable(ModelParametersMerger, BaseInstance)

local defaultSplitMode = "Accuracy"

local defaultMergeMode = "Average"

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

		if (#featureMatrix ~= #labelVector) then error("Feature matrix and the label vector does not contain the same number of rows!") end

	end

	self.featureMatrix = featureMatrix or self.featureMatrix

	self.labelVector = labelVector or self.labelVector

end

local function checkDepth(array, depth)

	depth = depth or 0

	local valueType = typeof(array)

	if (valueType == "table") then

		return checkDepth(array[1], depth + 1)

	else

		return depth

	end

end

local function checkIfIsTableOfMatrices(array)

	local depth = checkDepth(array)

	local isTableOfMatrices = (depth == 3)

	return isTableOfMatrices

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
	
	if (not Model) then error("No model!") end
	
	if (not modelType) then error("No model type!") end
	
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

local function getSplitAmountArrayFromAccuracyArray(splitMode, accuracyArray)
	
	local splitAmountArray
	
	local numberOfModelParameters = #accuracyArray
	
	if (splitMode == "Best") then
		
		local areAllZeroes = checkIfAllValuesAreZeroesInArray(accuracyArray)

		local bestModelParametersIndex

		if (areAllZeroes == true) then 

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

local function mergeModelParameters(mergeMode, ModelParametersArray, splitAmountArray)
	
	local NewModelParameters
	
	local numberOfModelParameters = #ModelParametersArray
	
	local isTableOfMatrices = checkIfIsTableOfMatrices(ModelParametersArray[1])
	
	if (isTableOfMatrices) and (mergeMode == "WeightedAverage") then

		NewModelParameters = calculateWeightedAverageModelParametersTable(ModelParametersArray, splitAmountArray)

	elseif (not isTableOfMatrices) and (mergeMode == "WeightedAverage") then

		NewModelParameters = calculateWeightedAverageModelParameters(ModelParametersArray, splitAmountArray)
		
	elseif (isTableOfMatrices)  and (mergeMode == "Average") then
		
		NewModelParameters = calculateAverageModelParametersTable(ModelParametersArray)
		
	elseif (not isTableOfMatrices) and (mergeMode == "Average") then
		
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
		
		local accuracyArray = generateAccuracyArray(self.Model, self.modelType, ModelParametersArray, self.featureMatrix, self.labelVector) 

		splitAmountArray = getSplitAmountArrayFromAccuracyArray(self.splitMode, accuracyArray)
		
	else
		
		warn("Using the existing split amount array.")
		
	end
	
	local NewModelParameters = mergeModelParameters(self.mergeMode, ModelParametersArray, splitAmountArray)

	return NewModelParameters

end

return ModelParametersMerger