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

local BaseModel = require(script.Parent.BaseModel)

MultinomialNaiveBayesModel = {}

MultinomialNaiveBayesModel.__index = MultinomialNaiveBayesModel

setmetatable(MultinomialNaiveBayesModel, BaseModel)

local defaultMode = "Hybrid"

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

	local ModelParameters = self.ModelParameters

	local ClassesList = self.ClassesList

	local initialProbability = (useLogProbabilities and 0) or 1

	local posteriorProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})

	for data, unwrappedFeatureVector in ipairs(featureMatrix) do

		featureVector = {unwrappedFeatureVector}

		label = labelVector[data][1]

		classIndex = table.find(ClassesList, label)

		featureProbabilityVector = {ModelParameters[1][classIndex]}

		priorProbabilityVector = {ModelParameters[2][classIndex]}

		posteriorProbabilityVector[data][1] = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

	end

	cost = self:logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

local function batchMultinomialNaiveBayes(extractedFeatureMatrixTable, numberOfData)
	
	local featureProbabilityMatrix = {}

	local priorProbabilityMatrix = {}

	local numberOfDataPointVector = {}

	local featureProbabilityVector

	local numberOfSubData

	local featureCountVector

	local sumFeatureCount
	
	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

		extractedFeatureMatrix = extractedFeatureMatrixTable[classIndex]

		numberOfSubData = #extractedFeatureMatrix

		featureCountVector = AqwamTensorLibrary:sum(extractedFeatureMatrix, 1)

		sumFeatureCount = AqwamTensorLibrary:sum(extractedFeatureMatrix)

		featureProbabilityVector = AqwamTensorLibrary:divide(featureCountVector, sumFeatureCount)

		featureProbabilityMatrix[classIndex] = featureProbabilityVector[1]

		priorProbabilityMatrix[classIndex] = {(numberOfSubData / numberOfData)}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return featureProbabilityMatrix, priorProbabilityMatrix, numberOfDataPointVector
	
end

local function sequentialMultinomialNaiveBayes(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityMatrix, numberOfDataPointVector)
	
	local newFeatureProbabilityMatrix = {}
	
	local newPriorProbabilityMatrix = {}
	
	local newNumberOfDataPointVector = {}
	
	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		
		
	end
	
	return newFeatureProbabilityMatrix, newPriorProbabilityMatrix, newNumberOfDataPointVector
	
end

local multinomialBayesFunctionList = {
	
	["Batch"] = batchMultinomialNaiveBayes,
	
	["Sequential"] = sequentialMultinomialNaiveBayes,
	
}

function MultinomialNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMultinomialNaiveBayesModel = BaseModel.new(parameterDictionary)

	setmetatable(NewMultinomialNaiveBayesModel, MultinomialNaiveBayesModel)
	
	NewMultinomialNaiveBayesModel:setName("MultinomialNaiveBayes")
	
	NewMultinomialNaiveBayesModel.mode = parameterDictionary.mode or defaultMode
	
	NewMultinomialNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)
		
		local mode = NewMultinomialNaiveBayesModel.mode
		
		local useLogProbabilities = NewMultinomialNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewMultinomialNaiveBayesModel.ModelParameters or {}

		local featureProbabilityMatrix = ModelParameters[1]

		local priorProbabilityMatrix = ModelParameters[2]
		
		local numberOfDataPointVector = ModelParameters[3]

		if (mode == "Hybrid") then

			mode = (featureProbabilityMatrix and priorProbabilityMatrix and numberOfDataPointVector and "Sequential") or "Batch"		

		end
		
		local multinomialBayesFunction = multinomialBayesFunctionList[mode]

		if (not multinomialBayesFunction) then error("Unknown mode.") end

		local numberOfData = #featureMatrix

		local extractedFeatureMatrixTable = NewMultinomialNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)
		
		if (useLogProbabilities) then

			if (featureProbabilityMatrix) then featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, featureProbabilityMatrix) end

			if (priorProbabilityMatrix) then priorProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityMatrix) end

		end
		
		featureProbabilityMatrix, priorProbabilityMatrix, numberOfDataPointVector = multinomialBayesFunction(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityMatrix, numberOfDataPointVector)
		
		if (useLogProbabilities) then

			featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, featureProbabilityMatrix)

			priorProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityMatrix)

		end

		NewMultinomialNaiveBayesModel.ModelParameters = {featureProbabilityMatrix, priorProbabilityMatrix, numberOfDataPointVector}

		local cost = NewMultinomialNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewMultinomialNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)
		
		local finalProbabilityVector

		local numberOfData = #featureMatrix

		local ClassesList = NewMultinomialNaiveBayesModel.ClassesList

		local useLogProbabilities = NewMultinomialNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewMultinomialNaiveBayesModel.ModelParameters

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

		return NewMultinomialNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewMultinomialNaiveBayesModel:setGenerateFunction(function(labelVector)

		local ClassesList = NewMultinomialNaiveBayesModel.ClassesList

		local ModelParameters = NewMultinomialNaiveBayesModel.ModelParameters

		local selectedFeatureProbabilityMatrix = {}

		local selectedPriorProbabilityMatrix = {}

		for data, unwrappedLabelVector in ipairs(labelVector) do

			local label = unwrappedLabelVector[1]

			local classIndex = table.find(ClassesList, label)

			if (classIndex) then

				selectedFeatureProbabilityMatrix[data] = ModelParameters[1][classIndex]

				selectedPriorProbabilityMatrix[data] = ModelParameters[2][classIndex]

			end

		end

		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selectedFeatureProbabilityMatrix)

		local noiseMatrix = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		local generatedFeatureMatrixPart1 = AqwamTensorLibrary:multiply(selectedStandardDeviationMatrix, noiseMatrix)

		local generatedFeatureMatrix = AqwamTensorLibrary:add(selectedMeanMatrix, generatedFeatureMatrixPart1)

		return generatedFeatureMatrix

	end)

	return NewMultinomialNaiveBayesModel

end

return MultinomialNaiveBayesModel
