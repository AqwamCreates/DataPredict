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

local BaseModel = require(script.Parent.BaseModel)

GaussianNaiveBayesModel = {}

GaussianNaiveBayesModel.__index = GaussianNaiveBayesModel

setmetatable(GaussianNaiveBayesModel, BaseModel)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

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

local function calculateGaussianProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector)

	local gaussianProbability = (useLogProbabilities and 0) or 1

	local exponentStep1Vector = AqwamTensorLibrary:subtract(featureVector, meanVector)

	local exponentStep2Vector = AqwamTensorLibrary:power(exponentStep1Vector, 2)

	local exponentPart3Vector = AqwamTensorLibrary:power(standardDeviationVector, 2)

	local exponentStep4Vector = AqwamTensorLibrary:divide(exponentStep2Vector, exponentPart3Vector)

	local exponentStep5Vector = AqwamTensorLibrary:multiply(-0.5, exponentStep4Vector)

	local exponentWithTermsVector = AqwamTensorLibrary:applyFunction(math.exp, exponentStep5Vector)

	local divisorVector = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local gaussianProbabilityVector = AqwamTensorLibrary:divide(exponentWithTermsVector, divisorVector)

	for column = 1, #gaussianProbabilityVector[1], 1 do

		if (useLogProbabilities) then

			gaussianProbability = gaussianProbability + gaussianProbabilityVector[1][column]

		else

			gaussianProbability = gaussianProbability * gaussianProbabilityVector[1][column]

		end

	end

	return gaussianProbability

end

local function calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityVector)

	local posteriorProbability

	local likelihoodProbability = calculateGaussianProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector)

	if (useLogProbabilities) then

		posteriorProbability = likelihoodProbability + priorProbabilityVector[1][1]

	else

		posteriorProbability = likelihoodProbability * priorProbabilityVector[1][1]

	end

	return posteriorProbability

end

function GaussianNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewGaussianNaiveBayesModel = BaseModel.new(parameterDictionary)

	setmetatable(NewGaussianNaiveBayesModel, GaussianNaiveBayesModel)
	
	NewGaussianNaiveBayesModel:setName("GaussianNaiveBayes")

	NewGaussianNaiveBayesModel.ClassesList = parameterDictionary.ClassesList or {}

	NewGaussianNaiveBayesModel.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, false)

	return NewGaussianNaiveBayesModel

end

function GaussianNaiveBayesModel:calculateCost(featureMatrix, labelVector)

	local cost
	
	local featureVector

	local meanVector

	local standardDeviationVector

	local priorProbabilityVector

	local posteriorProbability

	local probability

	local classIndex

	local label

	local numberOfData = #labelVector
	
	local useLogProbabilities = self.useLogProbabilities
	
	local predictedProbabilitiesMatrix = AqwamTensorLibrary:createTensor({numberOfData, #self.ClassesList})

	local posteriorProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})

	for data = 1, #featureMatrix, 1 do
		
		featureVector = {labelVector[data]}

		label = labelVector[data][1]

		classIndex = table.find(self.ClassesList, label)

		meanVector = {self.ModelParameters[1][classIndex]}

		standardDeviationVector = {self.ModelParameters[2][classIndex]}

		priorProbabilityVector = {self.ModelParameters[3][classIndex]}

		posteriorProbabilityVector[data][1] = calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityVector)

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

function GaussianNaiveBayesModel:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

end


function GaussianNaiveBayesModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	self:processLabelVector(labelVector)
	
	local cost

	local extractedFeatureMatrix

	local meanVector

	local standardDeviationVector
	
	local numberOfSubData

	local ModelParameters = self.ModelParameters

	local ClassesList = self.ClassesList

	local numberOfClasses = #ClassesList
	
	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local extractedFeatureMatricesTable = separateFeatureMatrixByClass(featureMatrix, labelVector, ClassesList)

	local meanMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

	local standardDeviationMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)
	
	local priorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, 1})

	for classIndex, classValue in ipairs(ClassesList) do

		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]
		
		numberOfSubData = #extractedFeatureMatrix

		standardDeviationVector, _, meanVector = AqwamTensorLibrary:standardDeviation(extractedFeatureMatrix, 1)

		meanMatrix[classIndex] = meanVector[1]

		standardDeviationMatrix[classIndex] = standardDeviationVector[1]

		priorProbabilityMatrix[classIndex] = {(numberOfSubData / numberOfData)}

	end

	if (ModelParameters) then

		meanMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[1], meanMatrix), 2) 

		standardDeviationMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[2], standardDeviationMatrix), 2) 

		priorProbabilityMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[3], priorProbabilityMatrix), 2) 

	end

	self.ModelParameters = {meanMatrix, standardDeviationMatrix, priorProbabilityMatrix}

	cost = self:calculateCost(featureMatrix, labelVector)

	return {cost}

end

function GaussianNaiveBayesModel:getLabelFromOutputMatrix(outputMatrix)

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

function GaussianNaiveBayesModel:predict(featureMatrix, returnOriginalOutput)

	local finalProbabilityVector

	local numberOfData = #featureMatrix

	local ClassesList = self.ClassesList
	
	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters

	local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

	for classIndex, classValue in ipairs(ClassesList) do

		local meanVector = {ModelParameters[1][classIndex]}

		local standardDeviationVector = {ModelParameters[2][classIndex]}

		local priorProbabilityVector = {ModelParameters[3][classIndex]}

		for i = 1, numberOfData, 1 do

			local featureVector = {featureMatrix[i]}

			posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityVector)

		end

	end

	if (returnOriginalOutput) then return posteriorProbabilityMatrix end

	return self:getLabelFromOutputMatrix(posteriorProbabilityMatrix)

end

function GaussianNaiveBayesModel:getClassesList()

	return self.ClassesList

end

function GaussianNaiveBayesModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

return GaussianNaiveBayesModel