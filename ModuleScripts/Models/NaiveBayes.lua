local BaseModel = require(script.Parent.BaseModel)

NaiveBayesModel = {}

NaiveBayesModel.__index = NaiveBayesModel

setmetatable(NaiveBayesModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local function getClassesList(labelVector)
	
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

local function extractFeatureMatrixFromPosition(featureMatrix, positionList)
	
	local extractedFeatureMatrix = {}
	
	for i = 1, #featureMatrix, 1 do
		
		if table.find(positionList, i) then
			
			table.insert(extractedFeatureMatrix, featureMatrix[i])
			
		end	
		
	end
	
	return extractedFeatureMatrix
	
end

local function separateFeatureMatrixByClass(featureMatrix, labelVector)
	
	local classesList = getClassesList(labelVector)
	
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
	
	return extractedFeatureMatricesTable, classesList
	
end

local function calculateGaussianDensity(featureMatrix, meanMatrix, standardDeviationMatrix)
	
	local exponentStep1 = AqwamMatrixLibrary:subtract(featureMatrix, meanMatrix)
	
	local exponentStep2 = AqwamMatrixLibrary:divide(exponentStep1, standardDeviationMatrix)
	
	local exponentStep3 = AqwamMatrixLibrary:multiply(exponentStep2, exponentStep2)
	
	local exponentStep4 = AqwamMatrixLibrary:multiply((-1/2), exponentStep3)
	
	local exponentWithTerms = AqwamMatrixLibrary:applyFunction(math.exp, exponentStep4)
	
	local fractionStep1 = AqwamMatrixLibrary:multiply(standardDeviationMatrix, math.sqrt(2 * math.pi))
	
	local fractionStep2 = AqwamMatrixLibrary:divide(1, fractionStep1)
	
	local gaussianDensity = AqwamMatrixLibrary:multiply(fractionStep2, exponentWithTerms)
	
	return gaussianDensity	
	
end

function NaiveBayesModel.new()
	
	local NewNaiveBayesModel = BaseModel.new()
	
	setmetatable(NewNaiveBayesModel, NaiveBayesModel)
	
	NewNaiveBayesModel.classesList = nil
	
	NewNaiveBayesModel.gaussianDensityMatrix = nil
	
	return NewNaiveBayesModel
	
end

function NaiveBayesModel:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end
	
	local extractedFeatureMatricesTable
	
	local gaussianDensityVector

	local extractedFeatureMatrix

	local featureVector

	local meanVector

	local standardDeviationVector

	local probabilitiesVector
	
	local meanMatrix
	
	local standardDeviationMatrix
	
	local probabilitiesMatrix
	
	local gaussianDensityMatrix
	
	extractedFeatureMatricesTable, self.classesList = separateFeatureMatrixByClass(featureMatrix, labelVector)
	
	if (self.ModelParameters) then
		
		meanMatrix = self.ModelParameters[1]
		
		standardDeviationMatrix = self.ModelParameters[2]
		
		probabilitiesMatrix = self.ModelParameters[3]
		
	else
		
		meanMatrix = AqwamMatrixLibrary:createMatrix(#self.classesList, #featureMatrix[1])
		
		standardDeviationMatrix = AqwamMatrixLibrary:createMatrix(#self.classesList, #featureMatrix[1])
		
		probabilitiesMatrix = AqwamMatrixLibrary:createMatrix(#self.classesList, #featureMatrix[1], 1)
		
	end
	
	if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
	
	for classIndex, classValue in ipairs(self.classesList) do
		
		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]
		
		meanVector = AqwamMatrixLibrary:verticalMean(extractedFeatureMatrix)
		
		standardDeviationVector = AqwamMatrixLibrary:verticalStandardDeviation(extractedFeatureMatrix)
		
		meanMatrix[classIndex] = meanVector[1]
		
		standardDeviationMatrix[classIndex] = standardDeviationVector[1]
		
		probabilitiesVector = AqwamMatrixLibrary:createMatrix(1, #featureMatrix[1], 1)
		
		for data = 1, #featureMatrix, 1 do
			
			featureVector = {featureMatrix[data]}
			
			gaussianDensityVector = calculateGaussianDensity(featureVector, meanVector, standardDeviationVector)
			
			probabilitiesVector = AqwamMatrixLibrary:multiply(probabilitiesVector, gaussianDensityVector)
			
		end
		
		probabilitiesMatrix[classIndex] = probabilitiesVector[1]
		
	end
	
	self.ModelParameters = {meanMatrix, standardDeviationMatrix, probabilitiesMatrix}
	
end

function NaiveBayesModel:predict(featureMatrix)
	
	local meanMatrix = self.ModelParameters[1]
	
	local standardDeviationMatrix = self.ModelParameters[2]
	
	local probabilitiesMatrix = self.ModelParameters[3]
	
	local priorProbabilitiesMatrix = calculateGaussianDensity(featureMatrix, meanMatrix, standardDeviationMatrix)
	
	local multipliedProbalitiesMatrices = AqwamMatrixLibrary:multiply(probabilitiesMatrix, priorProbabilitiesMatrix)
	
	local highestProbability = -math.huge
	
	local predictedClass
	
	local probabilityVector
	
	local probability
	
	for classIndex, classValue in ipairs(self.classesList) do
		
		probabilityVector = {multipliedProbalitiesMatrices[classIndex]}
		
		probability = 1
		
		for column = 1, #probabilityVector[1], 1 do
			
			probability *= probabilityVector[1][column]
			
		end
		
		if (probability > highestProbability) then
			
			predictedClass = classValue

			highestProbability = probability

		end
		
	end
	
	return predictedClass, highestProbability
	
end

return NaiveBayesModel
