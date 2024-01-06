local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ModelChecker = {}

ModelChecker.__index = ModelChecker

local defaultMaxNumberOfIterations = 100

local defaultMaxGeneralizationError = math.huge

local function calculateError(predictedLabelMatrix, trueLabelMatrix, numberOfData)
	
	local errorMatrix = AqwamMatrixLibrary:subtract(predictedLabelMatrix, trueLabelMatrix)

	errorMatrix = AqwamMatrixLibrary:power(errorMatrix, 2)

	local errorVector = AqwamMatrixLibrary:horizontalSum(errorMatrix)

	local totalError = AqwamMatrixLibrary:sum(errorVector)
	
	local calculatedError = totalError/numberOfData
	
	return calculatedError, errorVector
	
end

function ModelChecker.new(Model, modelType, maxNumberOfIterations, maxGeneralizationError)
	
	if (Model == nil) then error("No model in the ModelChecker!") end

	if (modelType == nil) then error("No model type in the ModelChecker!") end
	
	local NewModelChecker = {}
	
	setmetatable(NewModelChecker, ModelChecker)
	
	NewModelChecker.Model = Model
	
	NewModelChecker.modelType = modelType
	
	NewModelChecker.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewModelChecker.maxGeneralizationError = maxGeneralizationError or defaultMaxGeneralizationError
	
	return NewModelChecker
	
end

function ModelChecker:setParameters(Model, modelType, maxNumberOfIterations, maxGeneralizationError)
	
	self.Model = Model or self.Model
	
	self.modelType = modelType or self.modelType

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.maxGeneralizationError = maxGeneralizationError or self.maxGeneralizationError
	
end

function ModelChecker:setClassesList(classesList)

	self.ClassesList = classesList

end

function ModelChecker:convertLabelVectorToLogisticMatrix(labelVector)

	if (typeof(labelVector) == "number") then

		labelVector = {{labelVector}}

	end

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(#labelVector, #self.ClassesList)

	local label

	local labelPosition

	for row = 1, #labelVector, 1 do

		label = labelVector[row][1]

		labelPosition = table.find(self.ClassesList, label)

		logisticMatrix[row][labelPosition] = 1

	end

	return logisticMatrix

end

function ModelChecker:testClassification(testFeatureMatrix, testLabelVector)
	
	local testLabelMatrix
	
	if (#testLabelVector[1] == 1) then

		testLabelMatrix = self:convertLabelVectorToLogisticMatrix(testLabelVector)

	else

		testLabelMatrix = testLabelVector

	end
	
	local numberOfData = #testFeatureMatrix
	
	local predictedTestLabelMatrix = self.Model:predict(testFeatureMatrix, true)
	
	local calculatedError, errorVector = calculateError(predictedTestLabelMatrix, testLabelMatrix, numberOfData)

	return calculatedError, errorVector, predictedTestLabelMatrix

end

function ModelChecker:testRegression(testFeatureMatrix, testLabelVector)

	local numberOfData = #testFeatureMatrix
	
	local predictedLabelVector = self.Model:predict(testFeatureMatrix)

	local calculatedError, errorVector = calculateError(predictedLabelVector, testLabelVector, numberOfData)

	return calculatedError, errorVector, predictedLabelVector

end

function ModelChecker:validateClassification(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)

	local trainError

	local validationError
	
	local predictedTrainLabelMatrix
	
	local predictedValidationLabelMatrix
	
	local trainLabelMatrix
	
	local validationLabelMatrix
	
	local generalizationError
	
	local numberOfIterations = 0

	local trainErrorArray = {}

	local validationErrorArray = {}
	
	local numberOfValidationData = #validationFeatureMatrix
	
	local numberOfTrainData = #trainFeatureMatrix 
	
	if (#trainLabelVector[1] == 1) then

		trainLabelMatrix = self:convertLabelVectorToLogisticMatrix(trainLabelVector)

	else

		trainLabelMatrix = validationLabelVector

	end
	
	if (#validationLabelVector[1] == 1) then
		
		validationLabelMatrix = self:convertLabelVectorToLogisticMatrix(validationLabelVector)
		
	else
		
		validationLabelMatrix = validationLabelVector
		
	end

	repeat

		self.Model:train(trainFeatureMatrix, trainLabelMatrix)
		
		predictedTrainLabelMatrix = self.Model:predict(validationFeatureMatrix, true)
		
		predictedValidationLabelMatrix = self.Model:predict(validationFeatureMatrix, true)
		
		trainError = calculateError(predictedTrainLabelMatrix, trainLabelMatrix, numberOfValidationData)
	
		validationError = calculateError(predictedValidationLabelMatrix, validationLabelMatrix, numberOfValidationData)
		
		generalizationError = validationError - trainError

		table.insert(validationErrorArray, validationError)

		table.insert(trainErrorArray, trainError)

		numberOfIterations += 1

	until (numberOfIterations >= self.maxNumberOfIterations) or (generalizationError >= self.maxGeneralizationError)

	return trainErrorArray, validationErrorArray

end

function ModelChecker:validateRegression(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
	
	local trainError
	
	local validationError
	
	local validationCostVector
	
	local predictedLabelVector
	
	local generalizationError
	
	local numberOfIterations = 0

	local trainErrorArray = {}

	local validationErrorArray = {}
	
	local numberOfValidationData = #validationFeatureMatrix
	
	local predictedTrainLabelVector
	
	local predictedValidationLabelVector
	
	repeat
		
		self.Model:train(trainFeatureMatrix, trainLabelVector)

		predictedTrainLabelVector = self.Model:predict(validationFeatureMatrix)

		predictedValidationLabelVector = self.Model:predict(validationFeatureMatrix)

		trainError = calculateError(predictedTrainLabelVector, trainLabelVector, numberOfValidationData)

		validationError = calculateError(predictedValidationLabelVector, validationLabelVector, numberOfValidationData)

		generalizationError = validationError - trainError
		
		table.insert(trainErrorArray, trainError)

		table.insert(validationErrorArray, validationError)

		numberOfIterations += 1
		
	until (numberOfIterations >= self.maxNumberOfIterations) or (generalizationError >= self.maxGeneralizationError)
	
	return trainErrorArray, validationErrorArray
	
end

function ModelChecker:test(testFeatureMatrix, testLabelVector)
	
	local testCost

	local errorVector
	
	local predictedLabelMatrix

	if (self.modelType == "Regression") then

		testCost, errorVector, predictedLabelMatrix = self:testRegression(testFeatureMatrix, testLabelVector)

	elseif (self.modelType == "Classification") then

		testCost, errorVector, predictedLabelMatrix = self:testClassification(testFeatureMatrix, testLabelVector)
		
	else
		
		error("Invalid model type!")

	end

	return testCost, errorVector, predictedLabelMatrix

end

function ModelChecker:validate(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
	
	local trainCostArray
	
	local validationCostArray
	
	if (self.modelType == "Regression") then
		
		trainCostArray, validationCostArray = self:validateRegression(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
		
	elseif (self.modelType == "Classification") then
		
		trainCostArray, validationCostArray = self:validateClassification(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
		
	else
		
		error("Invalid model type!")
		
	end
	
	return trainCostArray, validationCostArray
	
end

return ModelChecker
