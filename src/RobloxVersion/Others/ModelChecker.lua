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
	
	local calculatedError = totalError/(2 * numberOfData)
	
	return calculatedError, errorVector
	
end

function ModelChecker.new(modelType, maxNumberOfIterations, maxGeneralizationError)
	
	local NewModelChecker = {}
	
	setmetatable(NewModelChecker, ModelChecker)
	
	NewModelChecker.modelType = modelType
	
	NewModelChecker.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewModelChecker.maxGeneralizationError = maxGeneralizationError or defaultMaxGeneralizationError
	
	NewModelChecker.Model = nil
	
	return NewModelChecker
	
end

function ModelChecker:setParameters(modelType, maxNumberOfIterations, maxGeneralizationError)
	
	self.modelType = modelType or self.modelType

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.maxGeneralizationError = maxGeneralizationError or self.maxGeneralizationError
	
end

function ModelChecker:setModel(Model)
	
	self.Model = Model
	
end

function ModelChecker:getModel()
	
	return self.Model
	
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
	
	local Model = self.Model

	if (not Model) then error("No model!") end
	
	local testLabelMatrix
	
	if (#testLabelVector[1] == 1) then

		testLabelMatrix = self:convertLabelVectorToLogisticMatrix(testLabelVector)

	else

		testLabelMatrix = testLabelVector

	end
	
	local numberOfData = #testFeatureMatrix
	
	local predictedTestLabelMatrix = Model:predict(testFeatureMatrix, true)
	
	local calculatedError, errorVector = calculateError(predictedTestLabelMatrix, testLabelMatrix, numberOfData)

	return calculatedError, errorVector, predictedTestLabelMatrix

end

function ModelChecker:testRegression(testFeatureMatrix, testLabelVector)
	
	local Model = self.Model

	if (not Model) then error("No model!") end

	local numberOfData = #testFeatureMatrix
	
	local predictedLabelVector = Model:predict(testFeatureMatrix)

	local calculatedError, errorVector = calculateError(predictedLabelVector, testLabelVector, numberOfData)

	return calculatedError, errorVector, predictedLabelVector

end

function ModelChecker:validateClassification(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
	
	local Model = self.Model
	
	if (not Model) then error("No model!") end

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
	
	local numberOfTrainData = #trainFeatureMatrix 
	
	local numberOfValidationData = #validationFeatureMatrix
	
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

		Model:train(trainFeatureMatrix, trainLabelMatrix)
		
		predictedTrainLabelMatrix = Model:predict(trainFeatureMatrix, true)
		
		predictedValidationLabelMatrix = Model:predict(validationFeatureMatrix, true)
		
		trainError = calculateError(predictedTrainLabelMatrix, trainLabelMatrix, numberOfTrainData)
	
		validationError = calculateError(predictedValidationLabelMatrix, validationLabelMatrix, numberOfValidationData)
		
		generalizationError = validationError - trainError

		table.insert(validationErrorArray, validationError)

		table.insert(trainErrorArray, trainError)

		numberOfIterations += 1

	until (numberOfIterations >= self.maxNumberOfIterations) or (generalizationError >= self.maxGeneralizationError)

	return trainErrorArray, validationErrorArray

end

function ModelChecker:validateRegression(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
	
	local Model = self.Model

	if (not Model) then error("No model!") end
	
	local trainError
	
	local validationError
	
	local validationCostVector
	
	local predictedLabelVector
	
	local generalizationError
	
	local numberOfIterations = 0

	local trainErrorArray = {}

	local validationErrorArray = {}
	
	local numberOfTrainData = #trainFeatureMatrix
	
	local numberOfValidationData = #validationFeatureMatrix
	
	local predictedTrainLabelVector
	
	local predictedValidationLabelVector
	
	repeat
		
		Model:train(trainFeatureMatrix, trainLabelVector)

		predictedTrainLabelVector = Model:predict(trainFeatureMatrix)

		predictedValidationLabelVector = Model:predict(validationFeatureMatrix)

		trainError = calculateError(predictedTrainLabelVector, trainLabelVector, numberOfTrainData)

		validationError = calculateError(predictedValidationLabelVector, validationLabelVector, numberOfValidationData)

		generalizationError = validationError - trainError
		
		table.insert(trainErrorArray, trainError)

		table.insert(validationErrorArray, validationError)

		numberOfIterations += 1
		
	until (numberOfIterations >= self.maxNumberOfIterations) or (generalizationError >= self.maxGeneralizationError)
	
	return trainErrorArray, validationErrorArray
	
end

function ModelChecker:test(testFeatureMatrix, testLabelVector)
	
	local calculatedError

	local errorVector
	
	local predictedLabelMatrix
	
	local modelType = self.modelType
	
	if (modelType == nil) then error("No model type!") end

	if (modelType == "Regression") then

		calculatedError, errorVector, predictedLabelMatrix = self:testRegression(testFeatureMatrix, testLabelVector)

	elseif (modelType == "Classification") then

		calculatedError, errorVector, predictedLabelMatrix = self:testClassification(testFeatureMatrix, testLabelVector)
		
	else
		
		error("Invalid model type!")

	end

	return calculatedError, errorVector, predictedLabelMatrix

end

function ModelChecker:validate(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
	
	local trainErrorArray
	
	local validationErrorArray
	
	local modelType = self.modelType
	
	if (modelType == "Regression") then
		
		trainErrorArray, validationErrorArray = self:validateRegression(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
		
	elseif (modelType == "Classification") then
		
		trainErrorArray, validationErrorArray = self:validateClassification(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
		
	else
		
		error("Invalid model type!")
		
	end
	
	return trainErrorArray, validationErrorArray
	
end

return ModelChecker
