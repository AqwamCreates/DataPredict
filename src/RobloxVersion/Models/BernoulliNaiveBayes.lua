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

local NaiveBayesBaseModel = require(script.Parent.NaiveBayesBaseModel)

BernoulliNaiveBayesModel = {}

BernoulliNaiveBayesModel.__index = BernoulliNaiveBayesModel

setmetatable(BernoulliNaiveBayesModel, NaiveBayesBaseModel)

local defaultMode = "Hybrid"

local function calculateBernoulliProbability(useLogProbabilities, featureVector, featureProbabilityVector)

	local bernoulliProbability = (useLogProbabilities and 0) or 1

	local functionToApply = function(featureValue, featureProbabilityValue) return (featureProbabilityValue * math.pow((1 - featureProbabilityValue), (1 - featureValue))) end

	local bernoulliProbabilityVector = AqwamTensorLibrary:applyFunction(functionToApply, featureVector, featureProbabilityVector)
	
	if (useLogProbabilities) then

		bernoulliProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, bernoulliProbabilityVector)

	end

	for column = 1, #bernoulliProbabilityVector[1], 1 do

		if (useLogProbabilities) then

			bernoulliProbability = bernoulliProbability + bernoulliProbabilityVector[1][column]

		else

			bernoulliProbability = bernoulliProbability * bernoulliProbabilityVector[1][column]

		end

	end

	return bernoulliProbability

end

local function calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

	local posteriorProbability

	local likelihoodProbability = calculateBernoulliProbability(useLogProbabilities, featureVector, featureProbabilityVector)

	if (useLogProbabilities) then

		posteriorProbability = likelihoodProbability + priorProbabilityVector[1][1]

	else

		posteriorProbability = likelihoodProbability * priorProbabilityVector[1][1]

	end

	return posteriorProbability

end

function BernoulliNaiveBayesModel:calculateCost(featureMatrix, labelVector)

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

		featureVector = {labelVector[data]}

		label = labelVector[data][1]

		classIndex = table.find(ClassesList, label)

		featureProbabilityVector = {ModelParameters[1][classIndex]}

		priorProbabilityVector = {ModelParameters[2][classIndex]}

		posteriorProbabilityVector[data][1] = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityVector)

	end

	cost = self:logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

local function batchBernoulliNaiveBayes(extractedFeatureMatrixTable, numberOfData)
	
	local featureProbabilityMatrix = {}

	local priorProbabilityVector = {}
	
	local numberOfDataPointVector = {}
	
	local numberOfSubData
	
	local featureProbabilityVector

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

		extractedFeatureMatrix = extractedFeatureMatrixTable[classIndex]

		numberOfSubData = #extractedFeatureMatrix

		featureProbabilityVector = AqwamTensorLibrary:mean(extractedFeatureMatrix, 1)

		featureProbabilityMatrix[classIndex] = featureProbabilityVector[1]

		priorProbabilityVector[classIndex] = {(numberOfSubData / numberOfData)}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}

	end
	
	return featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector
	
end

local function sequentialBernoulliNaiveBayes(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector)
	
	local sumMatrix = AqwamTensorLibrary:multiply(featureProbabilityMatrix, numberOfDataPointVector)

	local newTotalNumberOfDataPoint = numberOfData + AqwamTensorLibrary:sum(numberOfDataPointVector)

	local featureProbabilityMatrix = {}

	local priorProbabilityVector = {}

	local numberOfDataPointVector = {}
	
	local numberOfOldSubData

	local numberOfSubData
	
	local subSumVector
	
	local sumVector 

	local featureProbabilityVector

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		numberOfOldSubData = numberOfDataPointVector[classIndex][1]

		numberOfSubData = (#extractedFeatureMatrix + numberOfOldSubData)

		extractedFeatureMatrix = extractedFeatureMatrixTable[classIndex]
		
		subSumVector = AqwamTensorLibrary:sum(extractedFeatureMatrix, 1)

		sumVector = {sumMatrix[classIndex]}

		sumVector = AqwamTensorLibrary:add(sumVector, subSumVector)

		featureProbabilityVector = AqwamTensorLibrary:mean(sumVector, 1)

		featureProbabilityMatrix[classIndex] = featureProbabilityVector[1]

		priorProbabilityVector[classIndex] = {(numberOfSubData / newTotalNumberOfDataPoint)}

		numberOfDataPointVector[classIndex] = {numberOfSubData}

	end

	return featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector

end

local bernoulliNaiveBayesFunctionList = {
	
	["Batch"] = batchBernoulliNaiveBayes,
	
	["Sequential"] = sequentialBernoulliNaiveBayes,
	
}

function BernoulliNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewBernoulliNaiveBayes = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewBernoulliNaiveBayes, BernoulliNaiveBayesModel)
	
	NewBernoulliNaiveBayes:setName("BernoulliNaiveBayes")
	
	NewBernoulliNaiveBayes.mode = parameterDictionary.mode or defaultMode
	
	NewBernoulliNaiveBayes:setTrainFunction(function(featureMatrix, labelVector)
		
		local mode = NewBernoulliNaiveBayes.mode

		local useLogProbabilities = NewBernoulliNaiveBayes.useLogProbabilities

		local ModelParameters = NewBernoulliNaiveBayes.ModelParameters or {}

		local featureProbabilityMatrix = ModelParameters[1]

		local priorProbabilityVector = ModelParameters[2]

		local numberOfDataPointVector = ModelParameters[3]

		if (mode == "Hybrid") then

			mode = (featureProbabilityMatrix and priorProbabilityVector and numberOfDataPointVector and "Sequential") or "Batch"		

		end
		
		local bernoulliNaiveBayesFunction = bernoulliNaiveBayesFunctionList[mode]

		if (not bernoulliNaiveBayesFunction) then error("Unknown mode.") end

		local numberOfData = #featureMatrix

		local numberOfFeatures = #featureMatrix[1]

		local extractedFeatureMatrixTable = NewBernoulliNaiveBayes:separateFeatureMatrixByClass(featureMatrix, labelVector)

		if (mode == "Sequential") then

			local numberOfFeatures = #featureMatrix[1]

			local numberOfClasses = #NewBernoulliNaiveBayes.ClassesList

			local zeroValue = (useLogProbabilities and math.huge) or 0

			local oneValue = (useLogProbabilities and 0) or 1

			featureProbabilityMatrix = featureProbabilityMatrix or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, zeroValue)

			priorProbabilityVector = priorProbabilityVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, oneValue)

			numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)

		end
		
		if (useLogProbabilities) then

			if (featureProbabilityMatrix) then featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, featureProbabilityMatrix) end

			if (priorProbabilityVector) then priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityVector) end

		end
		
		featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector = bernoulliNaiveBayesFunction(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector)
		
		if (useLogProbabilities) then

			featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, featureProbabilityMatrix)

			priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)

		end

		NewBernoulliNaiveBayes.ModelParameters = {featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector}

		local cost = NewBernoulliNaiveBayes:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewBernoulliNaiveBayes:setPredictFunction(function(featureMatrix, returnOriginalOutput)
		
		local finalProbabilityVector

		local numberOfData = #featureMatrix

		local ClassesList = NewBernoulliNaiveBayes.ClassesList

		local useLogProbabilities = NewBernoulliNaiveBayes.useLogProbabilities

		local ModelParameters = NewBernoulliNaiveBayes.ModelParameters

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

		return NewBernoulliNaiveBayes:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewBernoulliNaiveBayes:setGenerateFunction(function(labelVector)

		local ClassesList = NewBernoulliNaiveBayes.ClassesList

		local ModelParameters = NewBernoulliNaiveBayes.ModelParameters

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

	return NewBernoulliNaiveBayes

end

return BernoulliNaiveBayesModel
