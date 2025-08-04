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

local BaseModel = require("Model_BaseModel")

ComplementNaiveBayesModel = {}

ComplementNaiveBayesModel.__index = ComplementNaiveBayesModel

setmetatable(ComplementNaiveBayesModel, BaseModel)

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

local function calculateComplementProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector)

	local complementProbability = (useLogProbabilities and 0) or 1

	local functionToApply = function(featureValue, complementFeatureProbabilityValue) return math.pow(complementFeatureProbabilityValue, featureValue) end

	local complementProbabilityVector = AqwamTensorLibrary:applyFunction(functionToApply, featureVector, complementFeatureProbabilityVector)
	
	if (useLogProbabilities) then
		
		complementProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, complementProbabilityVector)
		
	end

	for column = 1, #complementProbabilityVector[1], 1 do

		if (useLogProbabilities) then

			complementProbability = complementProbability + complementProbabilityVector[1][column]

		else

			complementProbability = complementProbability * complementProbabilityVector[1][column]

		end

	end

	return complementProbability

end

local function calculatePosteriorProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector, priorProbabilityVector)

	local posteriorProbability

	local likelihoodProbability = calculateComplementProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector)

	if (useLogProbabilities) then

		posteriorProbability = likelihoodProbability + priorProbabilityVector[1][1]

	else

		posteriorProbability = likelihoodProbability * priorProbabilityVector[1][1]

	end

	return posteriorProbability

end

function ComplementNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewComplementNaiveBayesModel = BaseModel.new(parameterDictionary)

	setmetatable(NewComplementNaiveBayesModel, ComplementNaiveBayesModel)
	
	NewComplementNaiveBayesModel:setName("ComplementNaiveBayes")

	NewComplementNaiveBayesModel.ClassesList = parameterDictionary.ClassesList or {}

	NewComplementNaiveBayesModel.useLogProbabilities = BaseModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, false)

	return NewComplementNaiveBayesModel

end

function ComplementNaiveBayesModel:calculateCost(featureMatrix, labelVector)

	local cost

	local featureVector

	local complementFeatureProbabilityVector

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

	local posteriorProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})

	for data = 1, #featureMatrix, 1 do

		featureVector = {labelVector[data]}

		label = labelVector[data][1]

		classIndex = table.find(self.ClassesList, label)

		complementFeatureProbabilityVector = {self.ModelParameters[1][classIndex]}

		priorProbabilityVector = {self.ModelParameters[2][classIndex]}

		posteriorProbabilityVector[data][1] = calculatePosteriorProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector, priorProbabilityVector)

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

function ComplementNaiveBayesModel:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

end


function ComplementNaiveBayesModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	self:processLabelVector(labelVector)

	local cost

	local extractedFeatureMatrix
	
	local extractedComplementFeatureMatrix
	
	local sumExtractedComplementFeatureVector
	
	local totalSumExtractedComplementFeatureVector
	
	local complementFeatureProbabilityVector

	local numberOfSubData
	
	local numberOfComplementSubData
	
	local totalNumberOfComplementSubData

	local ModelParameters = self.ModelParameters

	local ClassesList = self.ClassesList

	local numberOfClasses = #ClassesList

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local extractedFeatureMatricesTable = separateFeatureMatrixByClass(featureMatrix, labelVector, ClassesList)
	
	local complementFeatureProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

	local priorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, 1})

	for classIndex, classValue in ipairs(ClassesList) do

		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]

		numberOfSubData = #extractedFeatureMatrix
		
		totalSumExtractedComplementFeatureVector = nil
		
		totalNumberOfComplementSubData = 0
		
		for complementClassIndex, complementClassValue in ipairs(ClassesList) do
			
			if (complementClassIndex ~= classIndex) then
				
				extractedComplementFeatureMatrix = extractedFeatureMatricesTable[complementClassIndex]
				
				numberOfComplementSubData = #extractedComplementFeatureMatrix
				
				totalNumberOfComplementSubData = totalNumberOfComplementSubData + numberOfComplementSubData
				
				sumExtractedComplementFeatureVector = AqwamTensorLibrary:sum(extractedComplementFeatureMatrix, 1)
				
				if (totalSumExtractedComplementFeatureVector) then
					
					totalSumExtractedComplementFeatureVector = AqwamTensorLibrary:add(totalSumExtractedComplementFeatureVector, sumExtractedComplementFeatureVector)
					
				else
					
					totalSumExtractedComplementFeatureVector = sumExtractedComplementFeatureVector
					
				end
				
			end
			
		end
		
		complementFeatureProbabilityVector = AqwamTensorLibrary:divide(totalSumExtractedComplementFeatureVector, totalNumberOfComplementSubData)
		
		complementFeatureProbabilityMatrix[classIndex] = complementFeatureProbabilityVector[1]

		priorProbabilityMatrix[classIndex] = {(numberOfSubData / numberOfData)}

	end

	if (ModelParameters) then

		complementFeatureProbabilityMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[1], complementFeatureProbabilityMatrix), 2) 

		priorProbabilityMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[2], priorProbabilityMatrix), 2) 

	end

	self.ModelParameters = {complementFeatureProbabilityMatrix, priorProbabilityMatrix}

	cost = self:calculateCost(featureMatrix, labelVector)

	return {cost}

end

function ComplementNaiveBayesModel:getLabelFromOutputMatrix(outputMatrix)

	local numberOfData = #outputMatrix

	local predictedLabelVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestProbability

	local outputVector

	local classIndexArray

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		classIndexArray, highestProbability = AqwamTensorLibrary:findMinimumValueDimensionIndexArray(outputMatrix)

		if (classIndexArray == nil) then continue end

		predictedLabel = self.ClassesList[classIndexArray[2]]

		predictedLabelVector[i][1] = predictedLabel

		highestProbabilityVector[i][1] = highestProbability

	end

	return predictedLabelVector, highestProbabilityVector

end

function ComplementNaiveBayesModel:predict(featureMatrix, returnOriginalOutput)

	local finalProbabilityVector

	local numberOfData = #featureMatrix

	local ClassesList = self.ClassesList

	local useLogProbabilities = self.useLogProbabilities

	local ModelParameters = self.ModelParameters

	local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

	for classIndex, classValue in ipairs(ClassesList) do

		local complementFeatureProbabilityVector = {ModelParameters[1][classIndex]}

		local priorProbabilityVector = {ModelParameters[2][classIndex]}

		for i = 1, numberOfData, 1 do

			local featureVector = {featureMatrix[i]}

			posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector, priorProbabilityVector)

		end

	end

	if (returnOriginalOutput) then return posteriorProbabilityMatrix end

	return self:getLabelFromOutputMatrix(posteriorProbabilityMatrix)

end

function ComplementNaiveBayesModel:getClassesList()

	return self.ClassesList

end

function ComplementNaiveBayesModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

return ComplementNaiveBayesModel
