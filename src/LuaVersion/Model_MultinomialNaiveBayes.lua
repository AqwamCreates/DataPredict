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

local BaseModel = require("Model_BaseModel")

MultinomialNaiveBayesModel = {}

MultinomialNaiveBayesModel.__index = MultinomialNaiveBayesModel

setmetatable(MultinomialNaiveBayesModel, BaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local function extractFeatureMatrixFromPosition(featureMatrix, positionList)

	local extractedFeatureMatrix = {}

	for i = 1, #featureMatrix, 1 do

		if table.find(positionList, i) then

			table.insert(extractedFeatureMatrix, featureMatrix[i])

		end	

	end

	return extractedFeatureMatrix

end

local function separateFeatureMatrixByClass(featureMatrix, labelVector, classesList)

	local classesPositionTable = {}

	for classIndex, class in ipairs(classesList) do

		classesPositionTable[classIndex] = {}

		for i = 1, #labelVector, 1 do

			if (labelVector[i][1] == class) then

				table.insert(classesPositionTable[classIndex], i)

			end

		end

	end

	local extractedFeatureMatricesTable = {}

	local extractedFeatureMatrix

	for classIndex, class in ipairs(classesList) do

		extractedFeatureMatrix = extractFeatureMatrixFromPosition(featureMatrix, classesPositionTable[classIndex])

		table.insert(extractedFeatureMatricesTable, extractedFeatureMatrix)

	end

	return extractedFeatureMatricesTable

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

local function logLoss(labelVector, predictedProbabilitiesVector)

	local loglossFunction = function (y, p) return (y * math.log(p)) + ((1 - y) * math.log(1 - p)) end

	local logLossVector = AqwamTensorLibrary:applyFunction(loglossFunction, labelVector, predictedProbabilitiesVector)

	local logLossSum = AqwamTensorLibrary:sum(logLossVector)

	local logLoss = -logLossSum / #labelVector

	return logLoss

end

local function factorial(n)
	
	if (n > 1) then
		
		return factorial(n - 1)
		
	else
		
		return 1
		
	end
	
end

local function calculateMultinomialProbability(useLogProbabilities, featureVector, featureProbabilityVector)

	local multinomialProbabilityPart1 = (useLogProbabilities and 0) or 1
	
	if (useLogProbabilities) then
		
		featureProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, featureProbabilityVector)
		
	end

	for column = 1, #featureProbabilityVector[1], 1 do

		if (useLogProbabilities) then

			multinomialProbabilityPart1 = multinomialProbabilityPart1 + featureProbabilityVector[1][column]

		else

			multinomialProbabilityPart1 = multinomialProbabilityPart1 * featureProbabilityVector[1][column]

		end

	end
	
	local totalFeatureCount = AqwamTensorLibrary:sum(featureVector)
	
	local factorialSumFeatureCount = factorial(totalFeatureCount)
	
	local factorialFeatureVector = AqwamTensorLibrary:applyFunction(factorial, featureVector)
	
	local multipliedFactorialFeatureValue = 1
	
	for column = 1, #factorialFeatureVector[1], 1 do
		
		multipliedFactorialFeatureValue = multipliedFactorialFeatureValue * factorialFeatureVector[1][column]
		
	end
	
	local multinomialProbabilityPart2 = factorialSumFeatureCount / multipliedFactorialFeatureValue
	
	local multinomialProbability = multinomialProbabilityPart1 * multinomialProbabilityPart2

	return multinomialProbability

end

local function calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

	local posteriorProbability

	local likelihoodProbability = calculateMultinomialProbability(useLogProbabilities, featureVector, featureProbabilityVector)

	if (useLogProbabilities) then

		posteriorProbability = likelihoodProbability + priorProbabilityVector[1][1]

	else

		posteriorProbability = likelihoodProbability * priorProbabilityVector[1][1]

	end

	return posteriorProbability

end

function MultinomialNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMultinomialNaiveBayesModel = BaseModel.new(parameterDictionary)

	setmetatable(NewMultinomialNaiveBayesModel, MultinomialNaiveBayesModel)

	NewMultinomialNaiveBayesModel.ClassesList = parameterDictionary.ClassesList or {}

	NewMultinomialNaiveBayesModel.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, false)

	return NewMultinomialNaiveBayesModel

end

function MultinomialNaiveBayesModel:calculateCost(featureMatrix, labelVector)

	local cost

	local featureVector

	local featureProbabilityVector

	local priorProbabilityVector

	local posteriorProbability

	local probability

	local highestProbability

	local predictedClass

	local classIndex

	local label

	local numberOfData = #labelVector

	local useLogProbabilities = self.useLogProbabilities

	local initialProbability = (useLogProbabilities and 0) or 1

	local predictedProbabilitiesMatrix = AqwamTensorLibrary:createTensor({numberOfData, #self.ClassesList})

	local posteriorProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})

	for data = 1, #featureMatrix, 1 do

		featureVector = {labelVector[data]}

		label = labelVector[data][1]

		classIndex = table.find(self.ClassesList, label)

		featureProbabilityVector = {self.ModelParameters[1][classIndex]}

		priorProbabilityVector = {self.ModelParameters[2][classIndex]}

		posteriorProbabilityVector[data][1] = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

	end

	cost = logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

function MultinomialNaiveBayesModel:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

end


function MultinomialNaiveBayesModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	self:processLabelVector(labelVector)

	local cost

	local extractedFeatureMatrix

	local featureProbabilityVector

	local numberOfSubData
	
	local featureCountVector
	
	local sumFeatureCount

	local ModelParameters = self.ModelParameters

	local ClassesList = self.ClassesList

	local numberOfClasses = #ClassesList

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local extractedFeatureMatricesTable = separateFeatureMatrixByClass(featureMatrix, labelVector, ClassesList)

	local priorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, 1})

	local featureProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

	for classIndex, classValue in ipairs(ClassesList) do

		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]

		numberOfSubData = #extractedFeatureMatrix
		
		featureCountVector = AqwamTensorLibrary:sum(extractedFeatureMatrix, 1)

		sumFeatureCount = AqwamTensorLibrary:sum(extractedFeatureMatrix)
		
		featureProbabilityVector = AqwamTensorLibrary:divide(featureCountVector, sumFeatureCount)

		featureProbabilityMatrix[classIndex] = featureProbabilityVector[1]

		priorProbabilityMatrix[classIndex] = {(numberOfSubData / numberOfData)}

	end

	if (ModelParameters) then

		featureProbabilityMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[1], featureProbabilityMatrix), 2) 

		priorProbabilityMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[2], priorProbabilityMatrix), 2) 

	end

	self.ModelParameters = {featureProbabilityMatrix, priorProbabilityMatrix}

	cost = self:calculateCost(featureMatrix, labelVector)

	return {cost}

end

function MultinomialNaiveBayesModel:getLabelFromOutputMatrix(outputMatrix)

	local numberOfData = #outputMatrix

	local predictedLabelVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbability

	local outputVector

	local classIndexArray

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		classIndexArray, highestProbability = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(outputMatrix)

		if (classIndexArray == nil) then continue end

		predictedLabel = self.ClassesList[classIndexArray[2]]

		predictedLabelVector[i][1] = predictedLabel

		highestProbabilityVector[i][1] = highestProbability

	end

	return predictedLabelVector, highestProbabilityVector

end

function MultinomialNaiveBayesModel:predict(featureMatrix, returnOriginalOutput)

	local finalProbabilityVector

	local numberOfData = #featureMatrix

	local ClassesList = self.ClassesList

	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters

	local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

	for classIndex, classValue in ipairs(ClassesList) do

		local featureProbabilityVector = {ModelParameters[1][classIndex]}

		local priorProbabilityVector = {ModelParameters[2][classIndex]}

		for i = 1, numberOfData, 1 do

			local featureVector = {featureMatrix[i]}

			posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

		end

	end

	if (returnOriginalOutput) then return posteriorProbabilityMatrix end

	return self:getLabelFromOutputMatrix(posteriorProbabilityMatrix)

end

function MultinomialNaiveBayesModel:getClassesList()

	return self.ClassesList

end

function MultinomialNaiveBayesModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

return MultinomialNaiveBayesModel