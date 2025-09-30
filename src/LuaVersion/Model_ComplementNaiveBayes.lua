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

ComplementNaiveBayesModel = {}

ComplementNaiveBayesModel.__index = ComplementNaiveBayesModel

setmetatable(ComplementNaiveBayesModel, NaiveBayesBaseModel)

local defaultMode = "Hybrid"

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

function ComplementNaiveBayesModel:calculateCost(featureMatrix, labelVector)
	
	local useLogProbabilities = self.useLogProbabilities

	local ClassesList = self.ClassesList
	
	local ModelParameters = self.ModelParameters

	local complementFeatureProbabilityMatrix = ModelParameters[1]

	local priorProbabilityVector = ModelParameters[2]

	local posteriorProbabilityVector = {}

	local featureVector

	local complementFeatureProbabilityVector

	local priorProbabilityValue

	local posteriorProbabilityValue

	local classIndex

	local label

	for data, unwrappedFeatureVector in ipairs(featureMatrix) do

		featureVector = {unwrappedFeatureVector}

		label = labelVector[data][1]

		classIndex = table.find(ClassesList, label)
		
		if (classIndex) then
			
			complementFeatureProbabilityVector = {complementFeatureProbabilityMatrix[classIndex]}

			priorProbabilityValue = {priorProbabilityVector[classIndex]}
			
			posteriorProbabilityValue = calculatePosteriorProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector, priorProbabilityValue)
			
		else
			
			posteriorProbabilityValue = 0
			
		end

		posteriorProbabilityVector[data] = {posteriorProbabilityValue}

	end

	local cost = self:logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

local function batchComplementNaiveBayes(extractedFeatureMatrixTable, numberOfData, numberOfFeatures)
	
	local complementFeatureProbabilityMatrix = {}

	local priorProbabilityVector = {}
	
	local numberOfFeatureCountVector = {}

	local numberOfDataPointVector = {}

	local extractedFeatureMatrix

	local extractedComplementFeatureMatrix

	local sumExtractedComplementFeatureVector

	local totalSumExtractedComplementFeatureVector

	local complementFeatureProbabilityVector
	
	local numberOfFeatureCount

	local numberOfSubData

	local numberOfComplementSubData

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		if (extractedFeatureMatrix) then
			
			numberOfSubData = #extractedFeatureMatrix

			totalSumExtractedComplementFeatureVector = nil

			for complementClassIndex, extractedComplementFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

				if (complementClassIndex ~= classIndex) then

					numberOfComplementSubData = #extractedComplementFeatureMatrix

					sumExtractedComplementFeatureVector = AqwamTensorLibrary:sum(extractedComplementFeatureMatrix, 1)

					if (totalSumExtractedComplementFeatureVector) then

						totalSumExtractedComplementFeatureVector = AqwamTensorLibrary:add(totalSumExtractedComplementFeatureVector, sumExtractedComplementFeatureVector)

					else

						totalSumExtractedComplementFeatureVector = sumExtractedComplementFeatureVector

					end

				end

			end

			numberOfFeatureCount = AqwamTensorLibrary:sum(totalSumExtractedComplementFeatureVector)

			complementFeatureProbabilityVector = AqwamTensorLibrary:divide(totalSumExtractedComplementFeatureVector, numberOfFeatureCount)

			complementFeatureProbabilityMatrix[classIndex] = complementFeatureProbabilityVector[1]
			
		else

			numberOfSubData = 0
			
			numberOfFeatureCount = 0
			
			complementFeatureProbabilityMatrix[classIndex] = table.create(numberOfFeatures, 0)
			
		end
		
		numberOfFeatureCountVector[classIndex] = {numberOfFeatureCount}

		priorProbabilityVector[classIndex] = {(numberOfSubData / numberOfData)}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return complementFeatureProbabilityMatrix, priorProbabilityVector, numberOfFeatureCountVector, numberOfDataPointVector
	
end

local function sequentialComplementNaiveBayes(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, complementFeatureProbabilityMatrix, priorProbabilityVector, numberOfFeatureCountVector, numberOfDataPointVector)
	
	local newTotalNumberOfDataPoint = numberOfData + AqwamTensorLibrary:sum(numberOfDataPointVector)
	
	local newComplementFeatureProbabilityMatrix = {}
	
	local newNumberOfFeatureCountVector = {}
	
	local newNumberOfDataPointVector = {}
	
	local numberOfOldFeatureCount
	
	local numberOfFeatureCount
	
	local numberOfOldSubData

	local numberOfSubData
	
	local complementFeatureProbabilityVector
	
	local totalSumExtractedComplementFeatureVector
	
	local numberOfComplementSubData
	
	local sumExtractedComplementFeatureVector
	
	local newComplementFeatureProbabilityVector
	
	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

		numberOfOldSubData = numberOfDataPointVector[classIndex][1]
		
		if (extractedFeatureMatrix) then
			
			numberOfSubData = (#extractedFeatureMatrix + numberOfOldSubData)

			numberOfOldFeatureCount = numberOfFeatureCountVector[classIndex][1]

			complementFeatureProbabilityVector = {complementFeatureProbabilityMatrix[classIndex]}

			totalSumExtractedComplementFeatureVector = AqwamTensorLibrary:multiply(complementFeatureProbabilityVector, numberOfOldFeatureCount)

			for complementClassIndex, extractedComplementFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

				if (complementClassIndex ~= classIndex) then

					numberOfComplementSubData = #extractedComplementFeatureMatrix

					sumExtractedComplementFeatureVector = AqwamTensorLibrary:sum(extractedComplementFeatureMatrix, 1)

					totalSumExtractedComplementFeatureVector = AqwamTensorLibrary:add(totalSumExtractedComplementFeatureVector, sumExtractedComplementFeatureVector)

				end

			end

			numberOfFeatureCount = AqwamTensorLibrary:sum(totalSumExtractedComplementFeatureVector)

			newComplementFeatureProbabilityVector = AqwamTensorLibrary:divide(totalSumExtractedComplementFeatureVector, numberOfFeatureCount)

			newComplementFeatureProbabilityMatrix[classIndex] = newComplementFeatureProbabilityVector[1]
			
			newNumberOfFeatureCountVector[classIndex] = {numberOfFeatureCount}
			
		else
			
			numberOfSubData = numberOfOldSubData
			
			newComplementFeatureProbabilityMatrix[classIndex] = complementFeatureProbabilityMatrix[classIndex]
			
			newNumberOfFeatureCountVector[classIndex] = {numberOfOldFeatureCount}
			
		end

		newNumberOfDataPointVector[classIndex] = {numberOfSubData}

	end
	
	local newPriorProbabilityVector = AqwamTensorLibrary:divide(newNumberOfDataPointVector, newTotalNumberOfDataPoint)
	
	return newComplementFeatureProbabilityMatrix, newPriorProbabilityVector, newNumberOfFeatureCountVector, newNumberOfDataPointVector
	
end

local complementNaiveBayesFunctionList = {
	
	["Batch"] = batchComplementNaiveBayes,
	
	["Sequential"] = sequentialComplementNaiveBayes,
	
}

function ComplementNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewComplementNaiveBayesModel = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewComplementNaiveBayesModel, ComplementNaiveBayesModel)
	
	NewComplementNaiveBayesModel:setName("ComplementNaiveBayes")
	
	NewComplementNaiveBayesModel.mode = parameterDictionary.mode or defaultMode
	
	NewComplementNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)
		
		local mode = NewComplementNaiveBayesModel.mode
		
		local useLogProbabilities = NewComplementNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewComplementNaiveBayesModel.ModelParameters or {}

		local complementFeatureProbabilityMatrix = ModelParameters[1]

		local priorProbabilityVector = ModelParameters[2]
		
		local numberOfFeatureCountVector = ModelParameters[3]

		local numberOfDataPointVector = ModelParameters[4]

		if (mode == "Hybrid") then

			mode = (complementFeatureProbabilityMatrix and priorProbabilityVector and numberOfFeatureCountVector and numberOfDataPointVector and "Sequential") or "Batch"		

		end
		
		local complementNaiveBayesFunction = complementNaiveBayesFunctionList[mode]

		if (not complementNaiveBayesFunction) then error("Unknown mode.") end

		local numberOfData = #featureMatrix
		
		local numberOfFeatures = #featureMatrix[1]
		
		local extractedFeatureMatrixTable = NewComplementNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)
		
		if (mode == "Sequential") then

			local numberOfClasses = #NewComplementNaiveBayesModel.ClassesList

			local zeroValue = (useLogProbabilities and math.huge) or 0

			local oneValue = (useLogProbabilities and 0) or 1

			complementFeatureProbabilityMatrix = complementFeatureProbabilityMatrix or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, zeroValue)

			priorProbabilityVector = priorProbabilityVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, oneValue)

			numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)

		end
		
		if (useLogProbabilities) then

			if (complementFeatureProbabilityMatrix) then complementFeatureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, complementFeatureProbabilityMatrix) end

			if (priorProbabilityVector) then priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityVector) end

		end
		
		complementFeatureProbabilityMatrix, priorProbabilityVector, numberOfFeatureCountVector, numberOfDataPointVector = complementNaiveBayesFunction(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, complementFeatureProbabilityMatrix, priorProbabilityVector, numberOfFeatureCountVector, numberOfDataPointVector)
		
		if (useLogProbabilities) then

			complementFeatureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, complementFeatureProbabilityMatrix)

			priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)

		end

		NewComplementNaiveBayesModel.ModelParameters = {complementFeatureProbabilityMatrix, priorProbabilityVector, numberOfFeatureCountVector, numberOfDataPointVector}

		local cost = NewComplementNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewComplementNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)

		local ClassesList = NewComplementNaiveBayesModel.ClassesList

		local useLogProbabilities = NewComplementNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewComplementNaiveBayesModel.ModelParameters
		
		local complementFeatureProbabilityMatrix = ModelParameters[1]
		
		local priorProbabilityVector = ModelParameters[2]
		
		local numberOfData = #featureMatrix
		
		local numberOfClasses = #ClassesList

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClasses}, 0)

		for classIndex, classValue in ipairs(ClassesList) do

			local complementFeatureProbabilityVector = {complementFeatureProbabilityMatrix[classIndex]}

			local priorProbabilityValue = {priorProbabilityVector[classIndex]}

			for i = 1, numberOfData, 1 do

				local featureVector = {featureMatrix[i]}

				posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, complementFeatureProbabilityVector, priorProbabilityValue)

			end

		end

		if (returnOriginalOutput) then return posteriorProbabilityMatrix end

		return NewComplementNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewComplementNaiveBayesModel:setGenerateFunction(function(labelVector, totalCountVector)
		
		local numberOfData = #labelVector
		
		if (totalCountVector) then

			if (numberOfData ~= #totalCountVector) then error("The label vector and the total count does not contain the same number of rows.") end

		end
		
		local ClassesList = NewComplementNaiveBayesModel.ClassesList
		
		local useLogProbabilities = NewComplementNaiveBayesModel.useLogProbabilities
		
		local ModelParameters = NewComplementNaiveBayesModel.ModelParameters
		
		local complementFeatureProbabilityMatrix = ModelParameters[1]
		
		local numberOfFeatures = #complementFeatureProbabilityMatrix[1]
		
		local generatedFeatureMatrix = {}
		
		if (useLogProbabilities) then
			
			complementFeatureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, complementFeatureProbabilityMatrix)
			
		end
		
		totalCountVector = totalCountVector or AqwamTensorLibrary:createTensor({numberOfData, 1}, 1)
		
		local featureProbabilityMatrix = AqwamTensorLibrary:subtract(1, complementFeatureProbabilityMatrix)
		
		for data, unwrappedLabelVector in ipairs(labelVector) do

			local label = unwrappedLabelVector[1]

			local classIndex = table.find(ClassesList, label)

			if (classIndex) then

				local featureProbabilityArray = featureProbabilityMatrix[classIndex]

				local totalCount = totalCountVector[data][1]

				generatedFeatureMatrix[data] = sampleMultinomial(featureProbabilityArray, totalCount)

			else

				generatedFeatureMatrix[data] = table.create(numberOfFeatures, 0)

			end

		end

		return generatedFeatureMatrix
		
	end)

	return NewComplementNaiveBayesModel

end

return ComplementNaiveBayesModel
