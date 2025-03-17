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

NaiveBayesModel = {}

NaiveBayesModel.__index = NaiveBayesModel

setmetatable(NaiveBayesModel, BaseModel)

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

local function calculateGaussianDensity(useLogProbabilities, featureVector, meanVector, standardDeviationVector)
	
	local logGaussianDensity
	
	local exponentStep1 = AqwamTensorLibrary:subtract(featureVector, meanVector)
	
	local exponentStep2 = AqwamTensorLibrary:power(exponentStep1, 2)
	
	local exponentPart3 = AqwamTensorLibrary:power(standardDeviationVector, 2)
	
	local exponentStep4 = AqwamTensorLibrary:divide(exponentStep2, exponentPart3)
	
	local exponentStep5 = AqwamTensorLibrary:multiply(-0.5, exponentStep4)
	
	local exponentWithTerms = AqwamTensorLibrary:applyFunction(math.exp, exponentStep5)
	
	local divisor = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))
	
	local gaussianDensity = AqwamTensorLibrary:divide(exponentWithTerms, divisor)
	
	if (useLogProbabilities) then
		
		logGaussianDensity = AqwamTensorLibrary:applyFunction(math.log, gaussianDensity)
		
		return logGaussianDensity	
		
	else
		
		return gaussianDensity
		
	end
	
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

function NaiveBayesModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewNaiveBayesModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewNaiveBayesModel, NaiveBayesModel)
	
	NewNaiveBayesModel.ClassesList = parameterDictionary.ClassesList or {}
	
	NewNaiveBayesModel.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, false)
	
	return NewNaiveBayesModel
	
end

function NaiveBayesModel:calculateCost(featureMatrix, labelVector)
	
	local cost
	
	local featureVector
	
	local meanVector
	
	local standardDeviationVector
	
	local probabilitiesVector
	
	local priorProbabilitiesVector
	
	local initialProbability
	
	local multipliedProbalitiesVector
	
	local probability
	
	local highestProbability
	
	local predictedClass
	
	local classIndex
	
	local label
	
	local numberOfData = #labelVector
	
	local predictedProbabilitiesMatrix = AqwamTensorLibrary:createTensor({numberOfData, #self.ClassesList})
	
	local predictedProbabilitiesVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})
	
	if (self.useLogProbabilities) then

		initialProbability = 0

	else

		initialProbability = 1

	end
	
	for data = 1, #featureMatrix, 1 do
		
		featureVector = {featureMatrix[data]}
		
		label = labelVector[data][1]
		
		classIndex = table.find(self.ClassesList, label)
		
		meanVector = {self.ModelParameters[1][classIndex]}

		standardDeviationVector = {self.ModelParameters[2][classIndex]}

		probabilitiesVector = {self.ModelParameters[3][classIndex]}

		priorProbabilitiesVector = calculateGaussianDensity(self.useLogProbabilities, featureVector, meanVector, standardDeviationVector)

		multipliedProbalitiesVector = AqwamTensorLibrary:multiply(probabilitiesVector, priorProbabilitiesVector)
		
		probability = initialProbability
		
		for column = 1, #multipliedProbalitiesVector[1], 1 do

			if (self.useLogProbabilities) then

				probability += multipliedProbalitiesVector[1][column]

			else

				probability *= multipliedProbalitiesVector[1][column]

			end

		end
		
		predictedProbabilitiesVector[data][1] = probability

	end

	cost = logLoss(labelVector, predictedProbabilitiesVector)
	
	return {cost}
	
end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

function NaiveBayesModel:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector") end

	end

end

	
function NaiveBayesModel:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end
	
	self:processLabelVector(labelVector)
	
	local priorProbabilitiesVector

	local extractedFeatureMatrix

	local featureVector

	local meanVector

	local standardDeviationVector

	local probabilitiesVector
	
	local cost
	
	local ModelParameters = self.ModelParameters
	
	local ClassesList = self.ClassesList
	
	local numberOfClasses = #ClassesList
	
	local numberOfFeatures = #featureMatrix[1]
	
	local extractedFeatureMatricesTable = separateFeatureMatrixByClass(featureMatrix, labelVector, ClassesList)
	
	local meanMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

	local standardDeviationMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

	local probabilitiesMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 1)
	
	if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
	
	for classIndex, classValue in ipairs(self.ClassesList) do
		
		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]
		
		meanVector = AqwamTensorLibrary:mean(extractedFeatureMatrix, 1)
		
		standardDeviationVector = AqwamTensorLibrary:standardDeviation(extractedFeatureMatrix, 1)
		
		meanMatrix[classIndex] = meanVector[1]
		
		standardDeviationMatrix[classIndex] = standardDeviationVector[1]
		
		probabilitiesVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 1)
		
		for data = 1, #extractedFeatureMatrix, 1 do
			
			featureVector = {extractedFeatureMatrix[data]}
			
			priorProbabilitiesVector = calculateGaussianDensity(self.useLogProbabilities, featureVector, meanVector, standardDeviationVector)
			
			probabilitiesVector = AqwamTensorLibrary:multiply(probabilitiesVector, priorProbabilitiesVector)
			
		end
		
		probabilitiesMatrix[classIndex] = probabilitiesVector[1]
		
	end
	
	if (ModelParameters) then

		meanMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[1], meanMatrix), 2) 
		
		standardDeviationMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[2], standardDeviationMatrix), 2) 
		
		probabilitiesMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[3], probabilitiesMatrix), 2) 

	end
	
	self.ModelParameters = {meanMatrix, standardDeviationMatrix, probabilitiesMatrix}
	
	cost = self:calculateCost(featureMatrix, labelVector)
	
	return {cost}
	
end

function NaiveBayesModel:calculateFinalProbability(featureVector, probabilitiesVector, meanVector, standardDeviationVector)
	
	local finalProbability

	local priorProbabilitiesVector = calculateGaussianDensity(self.useLogProbabilities, featureVector, meanVector, standardDeviationVector)

	local multipliedProbalitiesVector = AqwamTensorLibrary:multiply(probabilitiesVector, priorProbabilitiesVector)

	if (self.useLogProbabilities) then

		finalProbability = AqwamTensorLibrary:sum(multipliedProbalitiesVector)

	else

		finalProbability = 1

		for column = 1, #multipliedProbalitiesVector[1], 1 do

			finalProbability *= multipliedProbalitiesVector[1][column]

		end

	end
	
	return finalProbability
	
end

function NaiveBayesModel:getLabelFromOutputMatrix(outputMatrix)
	
	local numberOfData = #outputMatrix

	local predictedLabelVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbabilitiesVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

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

		highestProbabilitiesVector[i][1] = highestProbability

	end

	return predictedLabelVector, highestProbabilitiesVector

end

function NaiveBayesModel:predict(featureMatrix, returnOriginalOutput)
	
	local finalProbabilityVector
	
	local numberOfData = #featureMatrix
	
	local ClassesList = self.ClassesList
	
	local ModelParameters = self.ModelParameters
	
	local finalProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)
	
	for classIndex, classValue in ipairs(ClassesList) do
		
		local meanVector = {ModelParameters[1][classIndex]}

		local standardDeviationVector = {ModelParameters[2][classIndex]}

		local probabilitiesVector = {ModelParameters[3][classIndex]}
		
		for i = 1, numberOfData, 1 do
			
			local featureVector = {featureMatrix[i]}
			
			finalProbabilityMatrix[i][classIndex] = self:calculateFinalProbability(featureVector, probabilitiesVector, meanVector, standardDeviationVector)
			
		end
		
	end
	
	if (returnOriginalOutput) then return finalProbabilityMatrix end
	
	return self:getLabelFromOutputMatrix(finalProbabilityMatrix)
	
end

function NaiveBayesModel:getClassesList()

	return self.ClassesList

end

function NaiveBayesModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

return NaiveBayesModel