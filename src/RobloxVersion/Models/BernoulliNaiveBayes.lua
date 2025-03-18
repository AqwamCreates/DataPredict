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

BernoulliNaiveBayesModel = {}

BernoulliNaiveBayesModel.__index = BernoulliNaiveBayesModel

setmetatable(BernoulliNaiveBayesModel, BaseModel)

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

local function calculateBernoulliProbability(useLogProbabilities, featureVector, featureProbabilityVector)
	
	local logBernoulliProbabilityVector
	
	local functionToApply = function(featureValue, featureProbabilityValue) return (featureProbabilityValue * math.pow((1 - featureProbabilityValue), (1 - featureValue))) end
	
	local bernoulliProbabilityVector = AqwamTensorLibrary:applyFunction(functionToApply, featureVector, featureProbabilityVector)
	
	if (useLogProbabilities) then
		
		logBernoulliProbabilityVector =  AqwamTensorLibrary:applyFunction(math.log, bernoulliProbabilityVector)
		
		return logBernoulliProbabilityVector
		
	else
		
		return bernoulliProbabilityVector
		
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

local function calculateFinalProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

	local finalProbability

	local likelihoodProbabilityVector = calculateBernoulliProbability(useLogProbabilities, featureVector, featureProbabilityVector)

	if (useLogProbabilities) then

		finalProbability = AqwamTensorLibrary:sum(likelihoodProbabilityVector)

		finalProbability = finalProbability + priorProbabilityVector[1][1]

	else

		finalProbability = 1

		for column = 1, #likelihoodProbabilityVector[1], 1 do

			finalProbability = finalProbability * likelihoodProbabilityVector[1][column]

		end

		finalProbability = finalProbability * priorProbabilityVector[1][1]

	end

	return finalProbability

end

function BernoulliNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewBernoulliNaiveBayes = BaseModel.new(parameterDictionary)

	setmetatable(NewBernoulliNaiveBayes, BernoulliNaiveBayesModel)

	NewBernoulliNaiveBayes.ClassesList = parameterDictionary.ClassesList or {}

	NewBernoulliNaiveBayes.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, false)

	return NewBernoulliNaiveBayes

end

function BernoulliNaiveBayesModel:calculateCost(featureMatrix, labelVector)

	local cost

	local featureVector

	local featureProbabilityVector

	local priorProbabilityVector

	local likelihoodProbabilityVector

	local probability

	local highestProbability

	local predictedClass

	local classIndex

	local label

	local numberOfData = #labelVector

	local useLogProbabilities = self.useLogProbabilities

	local initialProbability = (useLogProbabilities and 0) or 1

	local predictedProbabilitiesMatrix = AqwamTensorLibrary:createTensor({numberOfData, #self.ClassesList})

	local predictedProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})

	for data = 1, #featureMatrix, 1 do

		featureVector = {labelVector[data]}

		label = labelVector[data][1]

		classIndex = table.find(self.ClassesList, label)

		featureProbabilityVector = {self.ModelParameters[1][classIndex]}

		priorProbabilityVector = {self.ModelParameters[2][classIndex]}

		likelihoodProbabilityVector = calculateBernoulliProbability(useLogProbabilities, featureVector, featureProbabilityVector)

		probability = initialProbability

		for column = 1, #likelihoodProbabilityVector[1], 1 do

			if (useLogProbabilities) then

				probability = probability + likelihoodProbabilityVector[1][column]

			else

				probability = probability * likelihoodProbabilityVector[1][column]

			end

		end

		if (useLogProbabilities) then

			probability = probability + priorProbabilityVector[1][1]

		else

			probability = probability * priorProbabilityVector[1][1]

		end

		predictedProbabilityVector[data][1] = probability

	end

	cost = logLoss(labelVector, predictedProbabilityVector)

	return cost

end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

function BernoulliNaiveBayesModel:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

end


function BernoulliNaiveBayesModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	self:processLabelVector(labelVector)

	local cost

	local extractedFeatureMatrix

	local featureProbabilityVector

	local numberOfSubData

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
		
		featureProbabilityVector = AqwamTensorLibrary:mean(extractedFeatureMatrix, 1)
		
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

function BernoulliNaiveBayesModel:getLabelFromOutputMatrix(outputMatrix)

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

function BernoulliNaiveBayesModel:predict(featureMatrix, returnOriginalOutput)

	local finalProbabilityVector

	local numberOfData = #featureMatrix

	local ClassesList = self.ClassesList

	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters

	local finalProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

	for classIndex, classValue in ipairs(ClassesList) do

		local featureProbabilityVector = {ModelParameters[1][classIndex]}

		local priorProbabilityVector = {ModelParameters[2][classIndex]}

		for i = 1, numberOfData, 1 do

			local featureVector = {featureMatrix[i]}

			finalProbabilityMatrix[i][classIndex] = calculateFinalProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

		end

	end

	if (returnOriginalOutput) then return finalProbabilityMatrix end

	return self:getLabelFromOutputMatrix(finalProbabilityMatrix)

end

function BernoulliNaiveBayesModel:getClassesList()

	return self.ClassesList

end

function BernoulliNaiveBayesModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

return BernoulliNaiveBayesModel