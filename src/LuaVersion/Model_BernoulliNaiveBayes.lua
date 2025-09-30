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
	
	local useLogProbabilities = self.useLogProbabilities

	local ClassesList = self.ClassesList

	local ModelParameters = self.ModelParameters
	
	local featureProbabilityMatrix = ModelParameters[1]

	local priorProbabilityVector = ModelParameters[2]

	local posteriorProbabilityVector = {}

	local featureVector

	local featureProbabilityVector

	local priorProbabilityValue
	
	local posteriorProbabilityValue

	local classIndex

	local label

	for data, unwrappedFeatureVector in ipairs(featureMatrix) do

		featureVector = {unwrappedFeatureVector}

		label = labelVector[data][1]

		classIndex = table.find(ClassesList, label)
		
		if (classIndex) then
			
			featureProbabilityVector = {featureProbabilityMatrix[classIndex]}

			priorProbabilityValue = {priorProbabilityVector[classIndex]}
			
			posteriorProbabilityValue = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityValue)
			
		else
			
			posteriorProbabilityValue = 0
			
		end

		posteriorProbabilityVector[data] = {posteriorProbabilityValue}

	end
	
	if (useLogProbabilities) then
		
		posteriorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, posteriorProbabilityVector)
		
	end

	local cost = self:logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

local function batchBernoulliNaiveBayes(extractedFeatureMatrixTable, numberOfData, numberOfFeatures)
	
	local featureProbabilityMatrix = {}

	local priorProbabilityVector = {}
	
	local numberOfDataPointVector = {}
	
	local numberOfSubData
	
	local featureProbabilityVector

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		if (extractedFeatureMatrix) then
			
			numberOfSubData = #extractedFeatureMatrix

			featureProbabilityVector = AqwamTensorLibrary:mean(extractedFeatureMatrix, 1)

			featureProbabilityMatrix[classIndex] = featureProbabilityVector[1]
			
		else
			
			numberOfSubData = 0
			
			featureProbabilityMatrix[classIndex] = table.create(numberOfFeatures, 0)
			
		end

		priorProbabilityVector[classIndex] = {(numberOfSubData / numberOfData)}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}

	end
	
	return featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector
	
end

local function sequentialBernoulliNaiveBayes(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector)
	
	local sumMatrix = AqwamTensorLibrary:multiply(featureProbabilityMatrix, numberOfDataPointVector)

	local newTotalNumberOfDataPoint = numberOfData + AqwamTensorLibrary:sum(numberOfDataPointVector)

	local newFeatureProbabilityMatrix = {}

	local newNumberOfDataPointVector = {}
	
	local numberOfOldSubData

	local numberOfSubData
	
	local subSumVector
	
	local sumVector

	local featureProbabilityVector

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		numberOfOldSubData = numberOfDataPointVector[classIndex][1]
		
		if (extractedFeatureMatrix) then
			
			numberOfSubData = (#extractedFeatureMatrix + numberOfOldSubData)
			
			subSumVector = AqwamTensorLibrary:sum(extractedFeatureMatrix, 1)

			sumVector = {sumMatrix[classIndex]}

			sumVector = AqwamTensorLibrary:add(sumVector, subSumVector)

			featureProbabilityVector = AqwamTensorLibrary:divide(sumVector, numberOfSubData)
			
			newFeatureProbabilityMatrix[classIndex] = featureProbabilityVector[1]
			
		else
			
			numberOfSubData = numberOfOldSubData
			
			newFeatureProbabilityMatrix[classIndex] = featureProbabilityMatrix[classIndex]
			
		end
		
		newNumberOfDataPointVector[classIndex] = {numberOfSubData}

	end
	
	local newPriorProbabilityVector = AqwamTensorLibrary:divide(newNumberOfDataPointVector, newTotalNumberOfDataPoint)

	return newFeatureProbabilityMatrix, newPriorProbabilityVector, newNumberOfDataPointVector

end

local bernoulliNaiveBayesFunctionList = {
	
	["Batch"] = batchBernoulliNaiveBayes,
	
	["Sequential"] = sequentialBernoulliNaiveBayes,
	
}

function BernoulliNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewBernoulliNaiveBayesModel = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewBernoulliNaiveBayesModel, BernoulliNaiveBayesModel)
	
	NewBernoulliNaiveBayesModel:setName("BernoulliNaiveBayes")
	
	NewBernoulliNaiveBayesModel.mode = parameterDictionary.mode or defaultMode
	
	NewBernoulliNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)
		
		local mode = NewBernoulliNaiveBayesModel.mode

		local useLogProbabilities = NewBernoulliNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewBernoulliNaiveBayesModel.ModelParameters or {}

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

		local extractedFeatureMatrixTable = NewBernoulliNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)

		if (mode == "Sequential") then

			local numberOfClasses = #NewBernoulliNaiveBayesModel.ClassesList

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
		
		featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector = bernoulliNaiveBayesFunction(extractedFeatureMatrixTable, numberOfData, numberOfFeatures, featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector)
		
		if (useLogProbabilities) then

			featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.log, featureProbabilityMatrix)

			priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)

		end

		NewBernoulliNaiveBayesModel.ModelParameters = {featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector}

		local cost = NewBernoulliNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewBernoulliNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)

		local ClassesList = NewBernoulliNaiveBayesModel.ClassesList

		local useLogProbabilities = NewBernoulliNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewBernoulliNaiveBayesModel.ModelParameters
		
		local featureProbabilityMatrix = ModelParameters[1]
		
		local priorProbabilityVector = ModelParameters[2]
		
		local numberOfData = #featureMatrix
		
		local numberOfClasses = #ClassesList

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClasses}, 0)

		for classIndex, classValue in ipairs(ClassesList) do

			local featureProbabilityVector = {featureProbabilityMatrix[classIndex]}

			local priorProbabilityValue = {priorProbabilityVector[classIndex]}

			for i = 1, numberOfData, 1 do

				local featureVector = {featureMatrix[i]}

				posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, featureProbabilityVector, priorProbabilityValue)

			end

		end

		if (returnOriginalOutput) then return posteriorProbabilityMatrix end

		return NewBernoulliNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewBernoulliNaiveBayesModel:setGenerateFunction(function(labelVector, noiseMatrix)
		
		local numberOfData = #labelVector

		if (noiseMatrix) then

			if (numberOfData ~= #noiseMatrix) then error("The label vector and the noise matrix does not contain the same number of rows.") end

		end

		local ClassesList = NewBernoulliNaiveBayesModel.ClassesList

		local useLogProbabilities = NewBernoulliNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewBernoulliNaiveBayesModel.ModelParameters
		
		local featureProbabilityMatrix = ModelParameters[1]
		
		local numberOfFeatures = #featureProbabilityMatrix[1]
		
		local selectedFeatureProbabiltyMatrix = {}
		
		if (useLogProbabilities) then

			featureProbabilityMatrix = AqwamTensorLibrary:applyFunction(math.exp, featureProbabilityMatrix)

		end
		
		for data, unwrappedLabelVector in ipairs(labelVector) do

			local label = unwrappedLabelVector[1]

			local classIndex = table.find(ClassesList, label)

			if (classIndex) then
				
				selectedFeatureProbabiltyMatrix[data] = featureProbabilityMatrix[classIndex]
				
			else
				
				selectedFeatureProbabiltyMatrix[data] = table.create(numberOfFeatures, 0)

			end

		end
		
		noiseMatrix = noiseMatrix or AqwamTensorLibrary:createRandomUniformTensor({numberOfData, numberOfFeatures})
		
		local binaryProbabilityFunction = function(noiseProbability, featureProbability) return ((noiseProbability < featureProbability) and 1) or 0 end
		
		local generatedFeatureMatrix = AqwamTensorLibrary:applyFunction(binaryProbabilityFunction, noiseMatrix, selectedFeatureProbabiltyMatrix)

		return generatedFeatureMatrix

	end)

	return NewBernoulliNaiveBayesModel

end

return BernoulliNaiveBayesModel
