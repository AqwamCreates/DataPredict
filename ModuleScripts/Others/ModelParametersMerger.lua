local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local ModelParametersMerger = {}

ModelParametersMerger.__index = ModelParametersMerger

local defaultMergeType = "simpleAverage"

function ModelParametersMerger.new(Model, modelType, mergeType)
	
	if (Model == nil) then error("No models in the ModelParametersMerger!") end
	
	if (modelType == nil) then error("No modelType in the ModelParametersMerger!") end
	
	local NewModelParametersMerger = {}
	
	setmetatable(NewModelParametersMerger, ModelParametersMerger)
	
	NewModelParametersMerger.Model = Model
	
	NewModelParametersMerger.modelType = modelType
	
	NewModelParametersMerger.mergeType = mergeType or defaultMergeType
	
	NewModelParametersMerger.ModelParametersArray = {}
	
	NewModelParametersMerger.featureMatrix = nil
	
	NewModelParametersMerger.labelVector = nil
	
	return NewModelParametersMerger
	
end

function ModelParametersMerger:setParameters(Model, modelType, mergeType)
	
	self.Model = Model or self.mergeType
	
	self.modelType = modelType or self.modelType

	self.mergeType = mergeType or self.mergeType
	
end

function ModelParametersMerger:setModelParametersArray(ModelParametersArray)
	
	self.ModelParametersArray = ModelParametersArray
	
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
	
	if (typeof(array[1]) == "table") then
		
		return checkDepth(array[1], depth + 1)
		
	else
		
		return depth
		
	end
	
end

local function checkIfIsTable(array)
	
	local depth = checkDepth(array)
	
	local isTable = (depth == 2)
	
	return isTable
	
end

local function generateModelParametersTableWithMatricesOfZeroValues(ModelParameters)
	
	local NewModelParameters = {}
	
	for i, matrix in ipairs(ModelParameters) do
		
		local numberOfRows = #matrix
		
		local numberOfColumns = #matrix[1]
		
		local newMatrix = AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns)
		
		table.insert(newMatrix)
		
	end
	
	return NewModelParameters
	
end

local function applyFunctionToEachMatricesInModelParameters(functionToApply, ModelParameters)
	
	for k, matrix in ipairs(ModelParameters) do

		ModelParameters[k] =  AqwamMatrixLibrary:applyFunction(functionToApply, ModelParameters)

	end
	
	return ModelParameters
	
end

local function applyFunctionToModelParameters(functionToApply, ModelParametersArray)
	
	local NewModelParameters = generateModelParametersTableWithMatricesOfZeroValues(ModelParametersArray[1])

	for i, ModelParameters in ipairs(ModelParametersArray) do

		for j, matrix in ipairs(ModelParameters) do

			NewModelParameters[i] = AqwamMatrixLibrary:applyFunction(functionToApply, NewModelParameters[i], matrix)

		end

	end
	
	return NewModelParameters
	
end

local function simpleAverageMerge(ModelParametersArray)
	
	local NewModelParameters
	
	local numberOfModelParameters = #ModelParametersArray
	
	local isTable = checkIfIsTable(ModelParametersArray[1])
	
	if isTable then
		
		local averageFunction = function(x) return (x / numberOfModelParameters) end
		
		local addFunction = function(x, y) return (x + y) end
		
		NewModelParameters = applyFunctionToModelParameters(addFunction, ModelParametersArray)
		
		NewModelParameters = applyFunctionToEachMatricesInModelParameters(averageFunction, NewModelParameters)
		
	else
		
		NewModelParameters = AqwamMatrixLibrary:add(unpack(ModelParametersArray))
		
		NewModelParameters = AqwamMatrixLibrary:divide(NewModelParameters, numberOfModelParameters)
		
	end
	
	return NewModelParameters
	
end

local function calculateTotalFromArray(array)
	
	local total = 0
	
	for i, value in ipairs(array) do total += value end
	
	return total
	
end

local function convertValueArrayToPercentageArray(array)
	
	local total = calculateTotalFromArray(array)
	
	local percentageArray = {}
	
	for i, value in ipairs(array) do
		
		local percentage = (value / total)
		
		table.insert(percentageArray, percentage)
		
	end
	
	return percentageArray
	
end

local function calculateScaledModelParametersTable(ModelParametersArray, percentageArray)

	local NewModelParameters = generateModelParametersTableWithMatricesOfZeroValues(ModelParametersArray[1])

	for i, ModelParameters in ipairs(ModelParametersArray) do

		for j, matrix in ipairs(ModelParameters) do

			local calculatedMatrix = AqwamMatrixLibrary:multiply(matrix, percentageArray[i])

			NewModelParameters[i] = AqwamMatrixLibrary:add(NewModelParameters[i], calculatedMatrix)

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

local function weightedAverageMergeRegression(Model, ModelParametersArray, featureMatrix, labelVector)
	
	local errorArray = {}
	
	local errorPercentageArray
	
	local NewModelParameters
	
	for i, ModelParameters in ipairs(ModelParametersArray) do
		
		Model:setModelParameters(ModelParameters)
		
		local predictVector = Model:predict(featureMatrix)
		
		local errorVector = AqwamMatrixLibrary:subtract(labelVector, predictVector)
		
		local absoluteErrorVector = AqwamMatrixLibrary:apply(math.abs, errorVector)
		
		local errorValue = AqwamMatrixLibrary:sum(absoluteErrorVector)
		
		table.insert(errorArray, errorValue)
		
	end
	
	errorPercentageArray = convertValueArrayToPercentageArray(errorArray)
	
	NewModelParameters = calculateScaledModelParameters(ModelParametersArray, errorPercentageArray)
	
	return NewModelParameters
	
end

local function generateAccuracyArray(Model, ModelParametersArray, featureMatrix, labelVector)
	
	local accuracyArray = {}
	
	local totalLabel = #labelVector

	for i, ModelParameters in ipairs(ModelParametersArray) do

		local accuracy = 0

		local totalCorrect = 0

		Model:setModelParameters(ModelParameters)

		for j = 1, totalLabel, 1 do

			local label = Model:predict(featureMatrix)

			if (label == labelVector[j][1]) then

				totalCorrect += 1

			end

		end

		accuracy = totalCorrect / totalLabel

		table.insert(accuracyArray, accuracy)

	end
	
	return accuracyArray
	
end

local function weightedAverageMergeClassification(Model, ModelParametersArray, featureMatrix, labelVector)
	
	local isTable = checkIfIsTable(ModelParametersArray[1])

	local accuracyArray = generateAccuracyArray(Model, ModelParametersArray, featureMatrix, labelVector)
	
	local accuracyPercentageArray = convertValueArrayToPercentageArray(accuracyArray)
	
	local NewModelParameters
	
	if isTable then
		
		NewModelParameters = calculateScaledModelParametersTable(ModelParametersArray, accuracyPercentageArray)
		
	else
		
		NewModelParameters = calculateScaledModelParameters(ModelParametersArray, accuracyPercentageArray)
		
	end
	
	return NewModelParameters
	
end

local function weightedAverageMerge(Model, ModelParametersArray, modelType, featureMatrix, labelVector)
	
	local NewModelParameters

	local numberOfModelParameters = #ModelParametersArray
	
	if (modelType == "regression") then
		
		NewModelParameters =  weightedAverageMergeRegression(Model, ModelParametersArray, featureMatrix, labelVector)
		
	elseif (modelType == "classification") then
		
		NewModelParameters = weightedAverageMergeClassification(Model, ModelParametersArray, featureMatrix, labelVector)
		
	end

	return NewModelParameters
	
end

local function votingMerge(Model, ModelParametersArray, featureMatrix, labelVector)
	
	local NewModelParameters
	
	local highestNumberOfCorrects = -math.huge
	
	local numberOfCorrectsArray = {}
	
	local totalLabel = #labelVector

	for i, ModelParameters in ipairs(ModelParametersArray) do

		local numberOfCorrects = 0

		Model:setModelParameters(ModelParameters)

		for j = 1, totalLabel, 1 do

			local label = Model:predict(featureMatrix)

			if (label == labelVector[j][1]) then

				numberOfCorrects += 1

			end

		end

		table.insert(numberOfCorrectsArray, numberOfCorrects)

	end
	
	for i, numberOfCorrects in ipairs(numberOfCorrectsArray)  do
		
		if (numberOfCorrects > highestNumberOfCorrects) then 
			
			highestNumberOfCorrects = numberOfCorrects
			
			NewModelParameters = ModelParametersArray[i]
			
		end
		
	end
	
	return NewModelParameters
	
end

function ModelParametersMerger:generate()
	
	local Model = self.Model
	
	local mergeType = self.mergeType
	
	local modelType = self.modelType
	
	local featureMatrix = self.featureMatrix
	
	local labelVector = self.labelVector
	
	local ModelParametersArray = self.ModelParametersArray
	
	local NewModelParameters
	
	if (mergeType == "simpleAverage") then
		
		NewModelParameters = simpleAverageMerge(ModelParametersArray)
		
	elseif (mergeType == "weightedAverage") then
		
		NewModelParameters = weightedAverageMerge(Model, ModelParametersArray, modelType, featureMatrix, labelVector)
		
	elseif (mergeType == "voting") then
		
		NewModelParameters = votingMerge(Model, ModelParametersArray, featureMatrix, labelVector)
		
	end

	return NewModelParameters
	
end

return ModelParametersMerger
