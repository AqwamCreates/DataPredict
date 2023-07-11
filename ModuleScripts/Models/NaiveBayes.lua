local BaseModel = require(script.Parent.BaseModel)

NaiveBayesModel = {}

NaiveBayesModel.__index = NaiveBayesModel

setmetatable(NaiveBayesModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

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
	
	local exponentStep2 = AqwamMatrixLibrary:divide(exponentStep1, standardDeviationVector)
	
	local exponentStep3 = AqwamMatrixLibrary:multiply(exponentStep2, exponentStep2)
	
	local exponentStep4 = AqwamMatrixLibrary:multiply((-1/2), exponentStep3)
	
	local exponentWithTerms = AqwamMatrixLibrary:applyFunction(math.exp, exponentStep4)
	
	local fractionStep1 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))
	
	local fractionStep2 = AqwamMatrixLibrary:divide(1, fractionStep1)
	
	local gaussianDensity = AqwamMatrixLibrary:multiply(fractionStep2, exponentWithTerms)
	
	if (useLogProbabilities) then
		
		return logGaussianDensity	
		
	else
		
		logGaussianDensity = AqwamMatrixLibrary:applyFunction(math.log, gaussianDensity)
		
		return logGaussianDensity
		
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

	local labelVectorColumn = AqwamMatrixLibrary:transpose(labelVector)

	for i, value in ipairs(labelVectorColumn[1]) do

		if table.find(classesList, value) then continue end

		return true

	end

	return false

end

function NaiveBayesModel.new(useLogProbabilities)
	
	local NewNaiveBayesModel = BaseModel.new()
	
	setmetatable(NewNaiveBayesModel, NaiveBayesModel)
	
	NewNaiveBayesModel.ClassesList = {}
	
	NewNaiveBayesModel.UseLogProbabilities = BaseModel:getBooleanOrDefaultOption(useLogProbabilities, false)
	
	return NewNaiveBayesModel
	
end

function NaiveBayesModel:setParameters(useLogProbabilities)

	self.UseLogProbabilities = useLogProbabilities or self.UseLogProbabilities

end

function NaiveBayesModel:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end
	
	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		table.sort(self.ClassesList, function(a,b) return a < b end)

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the naive bayes\'s classes list is present in the label vector") end

	end
	
	local gaussianDensityVector

	local extractedFeatureMatrix

	local featureVector

	local meanVector

	local standardDeviationVector

	local probabilitiesVector
	
	local meanMatrix
	
	local standardDeviationMatrix
	
	local probabilitiesMatrix
	
	local extractedFeatureMatricesTable = separateFeatureMatrixByClass(featureMatrix, labelVector, self.ClassesList)
	
	if (self.ModelParameters) then
		
		meanMatrix = self.ModelParameters[1]
		
		standardDeviationMatrix = self.ModelParameters[2]
		
		probabilitiesMatrix = self.ModelParameters[3]
		
	else
		
		meanMatrix = AqwamMatrixLibrary:createMatrix(#self.ClassesList, #featureMatrix[1])
		
		standardDeviationMatrix = AqwamMatrixLibrary:createMatrix(#self.ClassesList, #featureMatrix[1])
		
		probabilitiesMatrix = AqwamMatrixLibrary:createMatrix(#self.ClassesList, #featureMatrix[1], 1)
		
	end
	
	if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
	
	for classIndex, classValue in ipairs(self.ClassesList) do
		
		extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]
		
		meanVector = AqwamMatrixLibrary:verticalMean(extractedFeatureMatrix)
		
		standardDeviationVector = AqwamMatrixLibrary:verticalStandardDeviation(extractedFeatureMatrix)
		
		meanMatrix[classIndex] = meanVector[1]
		
		standardDeviationMatrix[classIndex] = standardDeviationVector[1]
		
		probabilitiesVector = AqwamMatrixLibrary:createMatrix(1, #featureMatrix[1], 1)
		
		for data = 1, #featureMatrix, 1 do
			
			featureVector = {featureMatrix[data]}
			
			gaussianDensityVector = calculateGaussianDensity(self.UseLogProbabilities, featureVector, meanVector, standardDeviationVector)
			
			probabilitiesVector = AqwamMatrixLibrary:multiply(probabilitiesVector, gaussianDensityVector)
			
		end
		
		probabilitiesMatrix[classIndex] = probabilitiesVector[1]
		
	end
	
	self.ModelParameters = {meanMatrix, standardDeviationMatrix, probabilitiesMatrix}
	
end

function NaiveBayesModel:predict(featureMatrix)
	
	local meanVector 
	
	local standardDeviationVector
	
	local probabilitiesVector
	
	local priorProbabilitiesVector
	
	local multipliedProbalitiesVector 
	
	local highestProbability = -math.huge
	
	local predictedClass
	
	local probabilityVector
	
	local probability
	
	for classIndex, classValue in ipairs(self.ClassesList) do
		
		meanVector = {self.ModelParameters[1][classIndex]}

		standardDeviationVector = {self.ModelParameters[2][classIndex]}

		probabilitiesVector = {self.ModelParameters[3][classIndex]}
		
		priorProbabilitiesVector = calculateGaussianDensity(self.UseLogProbabilities, featureMatrix, meanVector, standardDeviationVector)
		
		multipliedProbalitiesVector = AqwamMatrixLibrary:multiply(probabilitiesVector, priorProbabilitiesVector)
		
		if (self.UseLogProbabilities) then
			
			probability = 0
			
		else
			
			probability = 1
			
		end
		
		for column = 1, #multipliedProbalitiesVector[1], 1 do
			
			if (self.UseLogProbabilities) then

				probability += multipliedProbalitiesVector[1][column]

			else

				probability *= multipliedProbalitiesVector[1][column]

			end
			
		end
		
		if (probability > highestProbability) then
			
			predictedClass = classValue

			highestProbability = probability

		end
		
	end
	
	return predictedClass, highestProbability
	
end

function NaiveBayesModel:getClassesList()

	return self.ClassesList

end

function NaiveBayesModel:setClassesList(classesList)

	self.ClassesList = classesList

end

return NaiveBayesModel
