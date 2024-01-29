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
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

NaiveBayesModel = {}

NaiveBayesModel.__index = NaiveBayesModel

setmetatable(NaiveBayesModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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
	
	local exponentStep1 = AqwamMatrixLibrary:subtract(featureVector, meanVector)
	
	local exponentStep2 = AqwamMatrixLibrary:power(exponentStep1, 2)
	
	local exponentPart3 = AqwamMatrixLibrary:power(standardDeviationVector, 2)
	
	local exponentStep4 = AqwamMatrixLibrary:divide(exponentStep2, exponentPart3)
	
	local exponentStep5 = AqwamMatrixLibrary:multiply(-0.5, exponentStep4)
	
	local exponentWithTerms = AqwamMatrixLibrary:applyFunction(math.exp, exponentStep5)
	
	local divisor = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))
	
	local gaussianDensity = AqwamMatrixLibrary:divide(exponentWithTerms, divisor)
	
	if (useLogProbabilities) then
		
		logGaussianDensity = AqwamMatrixLibrary:applyFunction(math.log, gaussianDensity)
		
		return logGaussianDensity	
		
	else
		
		return gaussianDensity
		
	end
	
end

local function createClassesList(labelVector)

	local classesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(classesList, value) then

			table.insert(classesList, value)

		end

	end

	return classesList

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, classesList)

	for i = 1, #labelVector, 1 do

		if table.find(classesList, labelVector[i][1]) then continue end

		return true

	end

	return false

end

local function logLoss(labelVector, predictedProbabilitiesVector)
	
	local loglossFunction = function (y, p) return (y * math.log(p)) + ((1 - y) * math.log(1 - p)) end
	
	local logLossVector = AqwamMatrixLibrary:applyFunction(loglossFunction, labelVector, predictedProbabilitiesVector)
	
	local logLossSum = AqwamMatrixLibrary:sum(logLossVector)

	local logLoss = -logLossSum / #labelVector
	
	return logLoss
	
end

function NaiveBayesModel.new(useLogProbabilities)
	
	local NewNaiveBayesModel = BaseModel.new()
	
	setmetatable(NewNaiveBayesModel, NaiveBayesModel)
	
	NewNaiveBayesModel.ClassesList = {}
	
	NewNaiveBayesModel.useLogProbabilities = BaseModel:getBooleanOrDefaultOption(useLogProbabilities, false)
	
	return NewNaiveBayesModel
	
end

function NaiveBayesModel:setParameters(useLogProbabilities)

	self.useLogProbabilities = useLogProbabilities or self.useLogProbabilities

end

function NaiveBayesModel:calculateCost(featureMatrix, labelVector)
	
	local cost
	
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
	
	local predictedProbabilitiesMatrix = AqwamMatrixLibrary:createMatrix(#labelVector, #self.ClassesList)
	
	local predictedProbabilitiesVector = AqwamMatrixLibrary:createMatrix(#labelVector, #labelVector[1])
	
	if (self.useLogProbabilities) then

		initialProbability = 0

	else

		initialProbability = 1

	end
	
	for data = 1, #featureMatrix, 1 do
		
		label = labelVector[data][1]
		
		classIndex = table.find(self.ClassesList, label)
		
		meanVector = {self.ModelParameters[1][classIndex]}

		standardDeviationVector = {self.ModelParameters[2][classIndex]}

		probabilitiesVector = {self.ModelParameters[3][classIndex]}

		priorProbabilitiesVector = calculateGaussianDensity(self.useLogProbabilities, featureMatrix, meanVector, standardDeviationVector)

		multipliedProbalitiesVector = AqwamMatrixLibrary:multiply(probabilitiesVector, priorProbabilitiesVector)
		
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
	
	return cost
	
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
	
	local gaussianDensityVector

	local extractedFeatureMatrix

	local featureVector

	local meanVector

	local standardDeviationVector

	local probabilitiesVector
	
	local meanMatrix
	
	local standardDeviationMatrix
	
	local probabilitiesMatrix
	
	local cost
	
	local extractedFeatureMatricesTable = separateFeatureMatrixByClass(featureMatrix, labelVector, self.ClassesList)
	
	meanMatrix = AqwamMatrixLibrary:createMatrix(#self.ClassesList, #featureMatrix[1])

	standardDeviationMatrix = AqwamMatrixLibrary:createMatrix(#self.ClassesList, #featureMatrix[1])

	probabilitiesMatrix = AqwamMatrixLibrary:createMatrix(#self.ClassesList, #featureMatrix[1], 1)
	
	if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
	
	for classIndex, classValue in ipairs(self.ClassesList) do
		
		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]
		
		meanVector = AqwamMatrixLibrary:verticalMean(extractedFeatureMatrix)
		
		standardDeviationVector = AqwamMatrixLibrary:verticalStandardDeviation(extractedFeatureMatrix)
		
		meanMatrix[classIndex] = meanVector[1]
		
		standardDeviationMatrix[classIndex] = standardDeviationVector[1]
		
		probabilitiesVector = AqwamMatrixLibrary:createMatrix(1, #featureMatrix[1], 1)
		
		for data = 1, #extractedFeatureMatrix, 1 do
			
			featureVector = {extractedFeatureMatrix[data]}
			
			gaussianDensityVector = calculateGaussianDensity(self.useLogProbabilities, featureVector, meanVector, standardDeviationVector)
			
			probabilitiesVector = AqwamMatrixLibrary:multiply(probabilitiesVector, gaussianDensityVector)
			
		end
		
		probabilitiesMatrix[classIndex] = probabilitiesVector[1]
		
	end
	
	if (self.ModelParameters) then

		meanMatrix = AqwamMatrixLibrary:divide(AqwamMatrixLibrary:add(self.ModelParameters[1], meanMatrix), 2) 
		
		standardDeviationMatrix = AqwamMatrixLibrary:divide(AqwamMatrixLibrary:add(self.ModelParameters[2], standardDeviationMatrix), 2) 
		
		probabilitiesMatrix = AqwamMatrixLibrary:divide(AqwamMatrixLibrary:add(self.ModelParameters[3], probabilitiesMatrix), 2) 

	end
	
	self.ModelParameters = {meanMatrix, standardDeviationMatrix, probabilitiesMatrix}
	
	cost = self:calculateCost(featureMatrix, labelVector)
	
	return {cost}
	
end

function NaiveBayesModel:calculateFinalProbability(featureVector, probabilitiesVector, meanVector, standardDeviationVector)
	
	local finalProbability

	local priorProbabilitiesVector = calculateGaussianDensity(self.useLogProbabilities, featureVector, meanVector, standardDeviationVector)

	local multipliedProbalitiesVector = AqwamMatrixLibrary:multiply(probabilitiesVector, priorProbabilitiesVector)

	if (self.useLogProbabilities) then

		finalProbability = AqwamMatrixLibrary:sum(multipliedProbalitiesVector)

	else

		finalProbability = 1

		for column = 1, #multipliedProbalitiesVector[1], 1 do

			finalProbability *= multipliedProbalitiesVector[1][column]

		end

	end
	
	return finalProbability
	
end

function NaiveBayesModel:getLabelFromOutputMatrix(outputMatrix)

	local predictedLabelVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestProbabilitiesVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local z = AqwamMatrixLibrary:applyFunction(math.exp, outputMatrix)

	local zSum = AqwamMatrixLibrary:horizontalSum(z)

	local softmaxMatrix = AqwamMatrixLibrary:divide(z, zSum)

	local highestProbability

	local softMaxVector

	local classIndex

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		softMaxVector = {softmaxMatrix[i]}

		highestProbability, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(softMaxVector)

		if (classIndex == nil) then continue end

		predictedLabel = self.ClassesList[classIndex[2]]

		predictedLabelVector[i][1] = predictedLabel

		highestProbabilitiesVector[i][1] = highestProbability

	end

	return predictedLabelVector, highestProbabilitiesVector

end

function NaiveBayesModel:predict(featureMatrix, returnOriginalOutput)
	
	local finalProbabilityVector
	
	local finalProbabilityMatrix = AqwamMatrixLibrary:createMatrix(#featureMatrix, #self.ClassesList)
	
	for classIndex, classValue in ipairs(self.ClassesList) do
		
		local meanVector = {self.ModelParameters[1][classIndex]}

		local standardDeviationVector = {self.ModelParameters[2][classIndex]}

		local probabilitiesVector = {self.ModelParameters[3][classIndex]}
		
		for i = 1, #featureMatrix, 1 do
			
			local featureVector = {featureMatrix[i]}
			
			finalProbabilityMatrix[i][classIndex] = self:calculateFinalProbability(featureVector, probabilitiesVector, meanVector, standardDeviationVector)
			
		end
		
	end
	
	if (returnOriginalOutput == true) then return finalProbabilityMatrix end
	
	return self:getLabelFromOutputMatrix(finalProbabilityMatrix)
	
end

function NaiveBayesModel:getClassesList()

	return self.ClassesList

end

function NaiveBayesModel:setClassesList(classesList)

	self.ClassesList = classesList

end

return NaiveBayesModel
