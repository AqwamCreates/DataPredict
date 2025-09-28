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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local NaiveBayesBaseModel = require("Model_NaiveBayesBaseModel")

MultinomialNaiveBayesModel = {}

MultinomialNaiveBayesModel.__index = MultinomialNaiveBayesModel

setmetatable(MultinomialNaiveBayesModel, NaiveBayesBaseModel)

local defaultMode = "Hybrid"

local function factorial(n)
	
	local value = 1
	
	for i = 2, n, 1 do
		
		value = value * i
		
	end
	
	return value
	
end

local function logFactorial(n)
	
	local value = 0

	for i = 2, n, 1 do

		value = value + math.log(i)

	end
	
	return value
	
end

local function sampleMultinomial(probabilityArray, totalCount)
	
	local numberOfProbabilities = #probabilityArray
	
	local remainingCount = totalCount
	
	local featureArray = {}
	
	for i, p in ipairs(probabilityArray) do
		
		local count = math.floor(p * totalCount + 0.5)
		
		featureArray[i] = count
		
		remainingCount = remainingCount - count
		
	end

	while (remainingCount > 0) do
		
		local idx = math.random(1, numberOfProbabilities)
		
		featureArray[idx] = featureArray[idx] + 1
		
		remainingCount = remainingCount - 1
		
	end
	
	return featureArray
end

local function calculateMultinomialProbability(useLogProbabilities, featureVector, featureProbabilityVector)

	local multinomialProbabilityPart1 = (useLogProbabilities and 0) or 1
	
	if (useLogProbabilities) then
		
		featureProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, featureProbabilityVector)
		
	end

	for column = 1, #featureProbabilityVector[1], 1 do
		
		if (useLogProbabilities) then
			
			multinomialProbabilityPart1 = multinomialProbabilityPart1 + featureVector[1][column] * featureProbabilityVector[1][column]
			
		else
			
			multinomialProbabilityPart1 = multinomialProbabilityPart1 * (featureProbabilityVector[1][column] ^ featureVector[1][column])
			
		end
		
	end
	
	local totalFeatureCount = AqwamTensorLibrary:sum(featureVector)
	
	local logFactorialSumFeatureCount = logFactorial(totalFeatureCount)
	
	local logFactorialFeatureVector = AqwamTensorLibrary:applyFunction(logFactorial, featureVector)
	
	local sumLogFactorialFeatureValue = 0
	
	for column = 1, #logFactorialFeatureVector[1], 1 do
		
		sumLogFactorialFeatureValue = sumLogFactorialFeatureValue + logFactorialFeatureVector[1][column]
		
	end
	
	local multinomialProbabilityPart2
	
	if (useLogProbabilities) then
		
		multinomialProbabilityPart2 = logFactorialSumFeatureCount - sumLogFactorialFeatureValue
		
	else
		
		multinomialProbabilityPart2 = math.exp(logFactorialSumFeatureCount - sumLogFactorialFeatureValue)
		
	end
	
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

	local priorProbabilityVector = {}
	
	local featureCountMatrix = {}

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

		priorProbabilityVector[classIndex] = {(numberOfSubData / numberOfData)}
		
		featureCountMatrix[classIndex] = featureCountVector[1]
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return featureProbabilityMatrix, priorProbabilityVector, featureCountMatrix, numberOfDataPointVector
	
end

local function sequentialMultinomialNaiveBayes(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityVector, featureCountMatrix, numberOfDataPointVector)
	
	local newFeatureProbabilityMatrix = {}
	
	local newPriorProbabilityVector = {}
	
	local newFeatureCountMatrix = {}
	
	local newNumberOfDataPointVector = {}
	
	local featureCountVector
	
	local oldFeatureCountVector
	
	local totalFeatureCountVector
	
	local sumFeatureCount
	
	local numberOfOldSubData
	
	local numberOfSubData
	
	local newTotalNumberOfDataPoint = numberOfData + AqwamTensorLibrary:sum(numberOfDataPointVector)
	
	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		featureCountVector = AqwamTensorLibrary:sum(extractedFeatureMatrix, 1)
		
		oldFeatureCountVector = {featureCountMatrix[classIndex]}
		
		totalFeatureCountVector = AqwamTensorLibrary:add(oldFeatureCountVector, featureCountVector)
		
		sumFeatureCount = AqwamTensorLibrary:sum(totalFeatureCountVector)
		
		newFeatureProbabilityMatrix[classIndex] = AqwamTensorLibrary:divide(totalFeatureCountVector, sumFeatureCount)[1]
		
		numberOfOldSubData = numberOfDataPointVector[classIndex][1]

		numberOfSubData = (#extractedFeatureMatrix + numberOfOldSubData)
		
		newPriorProbabilityVector[classIndex] = {(numberOfSubData / newTotalNumberOfDataPoint)}
		
		newFeatureCountMatrix[classIndex] = totalFeatureCountVector[1]

		newNumberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return newFeatureProbabilityMatrix, newPriorProbabilityVector, newFeatureCountMatrix, newNumberOfDataPointVector
	
end

local multinomialNaiveBayesFunctionList = {
	
	["Batch"] = batchMultinomialNaiveBayes,
	
	["Sequential"] = sequentialMultinomialNaiveBayes,
	
}

function MultinomialNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewMultinomialNaiveBayesModel = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewMultinomialNaiveBayesModel, MultinomialNaiveBayesModel)
	
	NewMultinomialNaiveBayesModel:setName("MultinomialNaiveBayes")
	
	NewMultinomialNaiveBayesModel.mode = parameterDictionary.mode or defaultMode
	
	NewMultinomialNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)
		
		local mode = NewMultinomialNaiveBayesModel.mode
		
		local useLogProbabilities = NewMultinomialNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewMultinomialNaiveBayesModel.ModelParameters or {}

		local featureProbabilityMatrix = ModelParameters[1]

		local priorProbabilityVector = ModelParameters[2]
		
		local featureCountMatrix = ModelParameters[3]
		
		local numberOfDataPointVector = ModelParameters[4]

		if (mode == "Hybrid") then

			mode = (featureProbabilityMatrix and priorProbabilityVector and featureCountMatrix and numberOfDataPointVector and "Sequential") or "Batch"		

		end
		
		local multinomialNaiveBayesFunction = multinomialNaiveBayesFunctionList[mode]

		if (not multinomialNaiveBayesFunction) then error("Unknown mode.") end
		
		if (mode == "Sequential") then

			local numberOfFeatures = #featureMatrix[1]

			local numberOfClasses = #NewMultinomialNaiveBayesModel.ClassesList

			local zeroValue = (useLogProbabilities and math.huge) or 0

			local oneValue = (useLogProbabilities and 0) or 1

			featureProbabilityMatrix = featureProbabilityMatrix or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, zeroValue)

			priorProbabilityVector = priorProbabilityVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, oneValue)

			numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)

		end

		local numberOfData = #featureMatrix

		local extractedFeatureMatrixTable = NewMultinomialNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)
		
		if (useLogProbabilities) then

			if (featureProbabilityMatrix) then featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, featureProbabilityMatrix) end

			if (priorProbabilityVector) then priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityVector) end

		end
		
		featureProbabilityMatrix, priorProbabilityVector, featureCountMatrix, numberOfDataPointVector = multinomialNaiveBayesFunction(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityVector, featureCountMatrix, numberOfDataPointVector)
		
		if (useLogProbabilities) then

			featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, featureProbabilityMatrix)

			priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)

		end

		NewMultinomialNaiveBayesModel.ModelParameters = {featureProbabilityMatrix, priorProbabilityVector, featureCountMatrix, numberOfDataPointVector}

		local cost = NewMultinomialNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewMultinomialNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)

		local numberOfData = #featureMatrix

		local ClassesList = NewMultinomialNaiveBayesModel.ClassesList

		local useLogProbabilities = NewMultinomialNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewMultinomialNaiveBayesModel.ModelParameters
		
		local featureProbabilityMatrix = ModelParameters[1]
		
		local priorProbabilityMatrix = ModelParameters[2]

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

		for classIndex, classValue in ipairs(ClassesList) do

			local featureProbabilityVector = {featureProbabilityMatrix[classIndex]}

			local priorProbabilityVector = {priorProbabilityMatrix[classIndex]}

			for i = 1, numberOfData, 1 do

				local featureVector = {featureMatrix[i]}

				posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

			end

		end

		if (returnOriginalOutput) then return posteriorProbabilityMatrix end

		return NewMultinomialNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewMultinomialNaiveBayesModel:setGenerateFunction(function(labelVector, totalCountVector)
		
		local numberOfData = #labelVector
		
		if (totalCountVector) then

			if (numberOfData ~= #totalCountVector) then error("The label vector and the total count vector does not contain the same number of rows.") end

		end

		local ClassesList = NewMultinomialNaiveBayesModel.ClassesList

		local ModelParameters = NewMultinomialNaiveBayesModel.ModelParameters
		
		local featureProbabiltyMatrix = ModelParameters[1]
		
		local generatedFeatureMatrix = {}
		
		totalCountVector = totalCountVector or AqwamTensorLibrary:createTensor({numberOfData, 1}, 1)

		for data, unwrappedLabelVector in ipairs(labelVector) do
			
			local label = unwrappedLabelVector[1]
			
			local classIndex = table.find(ClassesList, label)
			
			if (classIndex) then
				
				local featureProbabiltyArray = featureProbabiltyMatrix[classIndex]
				
				local totalCount = totalCountVector[data][1]
				
				generatedFeatureMatrix[data] = sampleMultinomial(featureProbabiltyArray, totalCount)
				
			end
			
		end

		return generatedFeatureMatrix

	end)

	return NewMultinomialNaiveBayesModel

end

return MultinomialNaiveBayesModel
