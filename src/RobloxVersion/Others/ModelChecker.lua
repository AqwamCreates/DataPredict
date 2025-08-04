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

local ModelChecker = {}

ModelChecker.__index = ModelChecker

setmetatable(ModelChecker, BaseInstance)

local defaultMaximumNumberOfIterations = 100

local defaultMaximumGeneralizationError = math.huge

local function calculateError(predictedLabelMatrix, trueLabelMatrix, numberOfData)
	
	local errorMatrix = AqwamTensorLibrary:subtract(predictedLabelMatrix, trueLabelMatrix)

	errorMatrix = AqwamTensorLibrary:power(errorMatrix, 2)

	local errorVector = AqwamTensorLibrary:sum(errorMatrix, 2)

	local totalError = AqwamTensorLibrary:sum(errorVector)
	
	local calculatedError = totalError/(2 * numberOfData)
	
	return calculatedError, errorVector
	
end

function ModelChecker.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewModelChecker = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewModelChecker, ModelChecker)
	
	NewModelChecker.modelType = parameterDictionary.modelType
	
	NewModelChecker.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	NewModelChecker.maximumGeneralizationError = parameterDictionary.maximumGeneralizationError or defaultMaximumGeneralizationError
	
	NewModelChecker.Model = parameterDictionary.Model
	
	return NewModelChecker
	
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

	local logisticMatrix = AqwamTensorLibrary:createTensor({#labelVector, #self.ClassesList}, 0)

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

		numberOfIterations = numberOfIterations + 1

	until (numberOfIterations >= self.maximumNumberOfIterations) or (generalizationError >= self.maximumGeneralizationError)

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

		numberOfIterations = numberOfIterations + 1
		
	until (numberOfIterations >= self.maximumNumberOfIterations) or (generalizationError >= self.maximumGeneralizationError)
	
	return trainErrorArray, validationErrorArray
	
end

function ModelChecker:test(testFeatureMatrix, testLabelVector)
	
	local calculatedError

	local errorVector
	
	local predictedLabelMatrix
	
	local modelType = self.modelType

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
