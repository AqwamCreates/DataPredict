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
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ModelParametersMerger = {}

ModelParametersMerger.__index = ModelParametersMerger

local defaultMergeType = "Average"

function ModelParametersMerger.new(Model, modelType, mergeType)

	local NewModelParametersMerger = {}

	setmetatable(NewModelParametersMerger, ModelParametersMerger)

	NewModelParametersMerger.Model = Model

	NewModelParametersMerger.modelType = modelType

	NewModelParametersMerger.mergeType = mergeType or defaultMergeType

	NewModelParametersMerger.ModelParametersArray = {}

	NewModelParametersMerger.featureMatrix = nil

	NewModelParametersMerger.labelVector = nil
	
	NewModelParametersMerger.customSplitPercentage = {}

	return NewModelParametersMerger

end

function ModelParametersMerger:setParameters(Model, modelType, mergeType)

	self.Model = Model or self.Model

	self.modelType = modelType or self.modelType

	self.mergeType = mergeType or self.mergeType

end

function ModelParametersMerger:setModelParameters(...)
	
	local inputtedModelParametersArray = {...}

	local proccesedModelsArray = ((#inputtedModelParametersArray > 0) and inputtedModelParametersArray) or nil
	
	self.ModelParametersArray = proccesedModelsArray

end

function ModelParametersMerger:setCustomSplitPercentageArray(splitPercentageArray)
	
	self.customSplitPercentage = splitPercentageArray or self.customSplitPercentage
	
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

local function checkIfIsTable(array)

	local depth = checkDepth(array)

	local isTable = (depth == 3)

	return isTable

end

local function generateModelParametersTableWithMatricesOfZeroValues(ModelParameters)

	local NewModelParameters = {}

	for i, matrix in ipairs(ModelParameters) do

		local numberOfRows = #matrix

		local numberOfColumns = #matrix[1]

		local newMatrix = AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns)

		table.insert(NewModelParameters, newMatrix)

	end

	return NewModelParameters

end

local function calculateTotalFromArray(array)

	local total = 0

	for i, value in ipairs(array) do total += value end

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

			percentage = (value / total)

		end

		table.insert(percentageArray, percentage)

	end
	
	return percentageArray

end

local function calculateScaledModelParametersTable(ModelParametersArray, percentageArray)

	local NewModelParameters = generateModelParametersTableWithMatricesOfZeroValues(ModelParametersArray[1])

	for i, ModelParameters in ipairs(ModelParametersArray) do

		for j, matrix in ipairs(ModelParameters) do

			local calculatedMatrix = AqwamMatrixLibrary:multiply(matrix, percentageArray[i])

			NewModelParameters[j] = AqwamMatrixLibrary:add(NewModelParameters[j], calculatedMatrix)

		end

	end

	return NewModelParameters

end

local function calculateScaledModelParameters(ModelParametersArray, percentageArray)

	local FirstModelParameters = ModelParametersArray[1]

	local NewModelParameters = AqwamMatrixLibrary:createMatrix(#FirstModelParameters, #FirstModelParameters[1])

	for j, percentage in ipairs(percentageArray) do

		local matrix = ModelParametersArray[j]

		local calculatedMatrix = AqwamMatrixLibrary:multiply(matrix, percentage)

		NewModelParameters = AqwamMatrixLibrary:add(NewModelParameters, calculatedMatrix)

	end

	return NewModelParameters

end

local function generateErrorArrayForRegression(Model, ModelParametersArray, featureMatrix, labelVector)

	local errorArray = {}

	for i, ModelParameters in ipairs(ModelParametersArray) do

		Model:setModelParameters(ModelParameters)

		local predictVector = Model:predict(featureMatrix)

		local errorVector = AqwamMatrixLibrary:subtract(labelVector, predictVector)

		local absoluteErrorVector = AqwamMatrixLibrary:applyFunction(math.abs, errorVector)

		local errorValue = AqwamMatrixLibrary:sum(absoluteErrorVector)

		table.insert(errorArray, errorValue)

	end

	return errorArray

end

local function generateErrorArrayForClustering(Model, ModelParametersArray, featureMatrix)

	local errorArray = {}

	for i, ModelParameters in ipairs(ModelParametersArray) do

		Model:setModelParameters(ModelParameters)

		local _, distanceVector = Model:predict(featureMatrix)

		local errorValue = AqwamMatrixLibrary:sum(distanceVector)

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

		if (allZeroes == false) then break end

	end

	return allZeroes

end

local function generateAccuracyForEachModel(Model, modelType, mergeType, ModelParametersArray, featureMatrix, labelVector)
	
	if (Model == nil) then error("No model!") end
	
	if (modelType == nil) then error("No model type!") end
	
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

local function getSplitPercentageArray(mergeType, accuracyArray)
	
	local percentageSplitArray
	
	local numberOfModelParameters = #accuracyArray
	
	if (mergeType == "Average") then
		
		percentageSplitArray = {}
	
	elseif (mergeType == "WeightedAverage") then

		percentageSplitArray = convertValueArrayToPercentageArray(accuracyArray)
		
	elseif (mergeType == "WeightedAverageEqual") then
		
		local average = 1 / numberOfModelParameters
		
		percentageSplitArray = table.create(numberOfModelParameters, average)

	elseif (mergeType == "Best") then

		local areAllZeroes = checkIfAllValuesAreZeroesInArray(accuracyArray)
		
		local bestModelParametersIndex

		if (areAllZeroes == true) then 
			
			bestModelParametersIndex = Random.new():NextInteger(1, numberOfModelParameters)
			
		else
			
			bestModelParametersIndex = getIndexOfHighestAccuracy(accuracyArray)

		end
		
		percentageSplitArray = table.create(numberOfModelParameters, 0)

		percentageSplitArray[bestModelParametersIndex] = 1

	else

		error("Invalid merge type!")

	end
	
	return percentageSplitArray
	
end

local function applyFunctionToEachMatricesInModelParameters(functionToApply, ModelParameters)

	for k, matrix in ipairs(ModelParameters) do

		ModelParameters[k] =  AqwamMatrixLibrary:applyFunction(functionToApply, matrix)

	end

	return ModelParameters

end

local function applyFunctionToModelParameters(functionToApply, ModelParametersArray)

	local NewModelParameters = generateModelParametersTableWithMatricesOfZeroValues(ModelParametersArray[1])

	for i, ModelParameters in ipairs(ModelParametersArray) do

		for j, matrix in ipairs(ModelParameters) do

			NewModelParameters[j] = AqwamMatrixLibrary:applyFunction(functionToApply, NewModelParameters[j], matrix)

		end

	end

	return NewModelParameters

end

local function mergeModelParameters(mergeType, ModelParametersArray, percentageSplitArray)
	
	local NewModelParameters
	
	local numberOfModelParameters = #ModelParametersArray
	
	local isTable = checkIfIsTable(ModelParametersArray[1])
	
	if (isTable) and (mergeType ~= "Average") then

		NewModelParameters = calculateScaledModelParametersTable(ModelParametersArray, percentageSplitArray)

	elseif (isTable == false) and (mergeType ~= "Average") then

		NewModelParameters = calculateScaledModelParameters(ModelParametersArray, percentageSplitArray)
		
	elseif (isTable) and (mergeType == "Average") then
		
		local averageFunction = function(x) return (x / numberOfModelParameters) end

		local addFunction = function(x, y) return (x + y) end

		NewModelParameters = applyFunctionToModelParameters(addFunction, ModelParametersArray)

		NewModelParameters = applyFunctionToEachMatricesInModelParameters(averageFunction, NewModelParameters)

	elseif (isTable == false) and (mergeType == "Average") then

		NewModelParameters = AqwamMatrixLibrary:add(table.unpack(ModelParametersArray))

		NewModelParameters = AqwamMatrixLibrary:divide(NewModelParameters, numberOfModelParameters)

	end
	
	return NewModelParameters
	
end

function ModelParametersMerger:generate()

	local Model = self.Model

	local modelType = self.modelType

	local mergeType = self.mergeType

	local featureMatrix = self.featureMatrix

	local labelVector = self.labelVector

	local ModelParametersArray = self.ModelParametersArray

	if (typeof(ModelParametersArray) ~= "table") then error("No model parameters set!") end

	local percentageSplitArray

	if (mergeType == "Custom") then

		percentageSplitArray = self.customSplitPercentage

	elseif (mergeType ~= "Average") then
		
		local accuracyArray = generateAccuracyForEachModel(Model, modelType, mergeType, ModelParametersArray, featureMatrix, labelVector) 

		percentageSplitArray = getSplitPercentageArray(mergeType, accuracyArray)

	end

	local NewModelParameters = mergeModelParameters(mergeType, ModelParametersArray, percentageSplitArray)

	return NewModelParameters

end

return ModelParametersMerger
