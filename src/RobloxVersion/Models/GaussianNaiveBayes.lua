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

GaussianNaiveBayesModel = {}

GaussianNaiveBayesModel.__index = GaussianNaiveBayesModel

setmetatable(GaussianNaiveBayesModel, NaiveBayesBaseModel)

local defaultMode = "Hybrid"

local function calculateGaussianProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector)

	local gaussianProbability = (useLogProbabilities and 0) or 1

	local exponentStep1Vector = AqwamTensorLibrary:subtract(featureVector, meanVector)

	local exponentStep2Vector = AqwamTensorLibrary:power(exponentStep1Vector, 2)

	local exponentPart3Vector = AqwamTensorLibrary:power(standardDeviationVector, 2)

	local exponentStep4Vector = AqwamTensorLibrary:divide(exponentStep2Vector, exponentPart3Vector)

	local exponentStep5Vector = AqwamTensorLibrary:multiply(-0.5, exponentStep4Vector)

	local exponentWithTermsVector = AqwamTensorLibrary:applyFunction(math.exp, exponentStep5Vector)

	local divisorVector = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local gaussianProbabilityVector = AqwamTensorLibrary:divide(exponentWithTermsVector, divisorVector)

	for column = 1, #gaussianProbabilityVector[1], 1 do

		if (useLogProbabilities) then

			gaussianProbability = gaussianProbability + gaussianProbabilityVector[1][column]

		else

			gaussianProbability = gaussianProbability * gaussianProbabilityVector[1][column]

		end

	end

	return gaussianProbability

end

local function calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityVector)

	local posteriorProbability

	local likelihoodProbability = calculateGaussianProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector)

	if (useLogProbabilities) then

		posteriorProbability = likelihoodProbability + priorProbabilityVector[1][1]

	else

		posteriorProbability = likelihoodProbability * priorProbabilityVector[1][1]

	end

	return posteriorProbability

end

function GaussianNaiveBayesModel:calculateCost(featureMatrix, labelVector)

	local useLogProbabilities = self.useLogProbabilities
	
	local ClassesList = self.ClassesList
	
	local ModelParameters = self.ModelParameters
	
	local meanMatrix = ModelParameters[1]

	local standardDeviationMatrix = ModelParameters[2]

	local priorProbabilityVector = ModelParameters[3]

	local posteriorProbabilityVector = {}

	local featureVector

	local meanVector

	local standardDeviationVector

	local priorProbabilityValue
	
	local posteriorProbabilityValue

	local classIndex

	local label
	
	for data, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		featureVector = {unwrappedFeatureVector}

		label = labelVector[data][1]

		classIndex = table.find(ClassesList, label)
		
		if (classIndex) then
			
			meanVector = {meanMatrix[classIndex]}

			standardDeviationVector = {standardDeviationMatrix[classIndex]}

			priorProbabilityValue = {priorProbabilityVector[classIndex]}

			posteriorProbabilityValue = calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityValue)
			
		else
			
			posteriorProbabilityValue = 0
			
		end
		
		posteriorProbabilityVector[data] = {posteriorProbabilityValue}
		
	end

	local cost = self:logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

local function batchGaussianNaiveBayes(extractedFeatureMatrixTable, numberOfData)
	
	local numberOfClasses = #extractedFeatureMatrixTable
	
	local meanMatrix = {}

	local standardDeviationMatrix = {}

	local priorProbabilityVector = {}
	
	local numberOfDataPointVector = {}
	
	local extractedFeatureMatrix
	
	local standardDeviationVector
	
	local meanVector
	
	local numberOfSubData
	
	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		numberOfSubData = #extractedFeatureMatrix

		standardDeviationVector, _, meanVector = AqwamTensorLibrary:standardDeviation(extractedFeatureMatrix, 1)

		meanMatrix[classIndex] = meanVector[1]

		standardDeviationMatrix[classIndex] = standardDeviationVector[1]

		priorProbabilityVector[classIndex] = {(numberOfSubData / numberOfData)}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return meanMatrix, standardDeviationMatrix, priorProbabilityVector, numberOfDataPointVector
	
end

local function sequentialGaussianNaiveBayes(extractedFeatureMatrixTable, numberOfData, meanMatrix, standardDeviationMatrix, priorProbabilityVector, numberOfDataPointVector)
	
	local sumMatrix = AqwamTensorLibrary:multiply(meanMatrix, numberOfDataPointVector)

	local varianceMatrix = AqwamTensorLibrary:power(standardDeviationMatrix, 2)

	local multipliedVarianceMatrix = AqwamTensorLibrary:multiply(varianceMatrix, numberOfDataPointVector)

	local newTotalNumberOfDataPoint = numberOfData + AqwamTensorLibrary:sum(numberOfDataPointVector)

	local newMeanMatrix = {}

	local newStandardDeviationVector = {}

	local newPriorProbabilityVector = {}

	local newNumberOfDataPointVector = {}
	
	local extractedFeatureMatrix
	
	local numberOfOldSubData
	
	local numberOfSubData
	
	local subSumVector
	
	local sumVector
	
	local oldMeanVector
	
	local newMeanVector
	
	local featureMatrixMinusMeanMatrix

	local featureMatrixMinusNewMeanMatrix

	local multipliedAdjustedFeatureMatrix

	local subMultipliedVarianceVector

	local multipliedVarianceVector

	local newVarianceVector
	
	local newStandardDeviationVector

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		numberOfOldSubData = numberOfDataPointVector[classIndex][1]
		
		numberOfSubData = (#extractedFeatureMatrix + numberOfOldSubData)
		
		subSumVector = AqwamTensorLibrary:sum(extractedFeatureMatrix, 1)
		
		sumVector = {sumMatrix[classIndex]}
		
		sumVector = AqwamTensorLibrary:add(sumVector, subSumVector)
		
		oldMeanVector = {meanMatrix[classIndex]}
		
		newMeanVector = AqwamTensorLibrary:divide(sumVector, numberOfSubData)
		
		-- Welford's algorithm for calculating new variance.
		
		featureMatrixMinusMeanMatrix = AqwamTensorLibrary:subtract(extractedFeatureMatrix, oldMeanVector)
		
		featureMatrixMinusNewMeanMatrix = AqwamTensorLibrary:subtract(extractedFeatureMatrix, newMeanVector)
		
		multipliedAdjustedFeatureMatrix = AqwamTensorLibrary:multiply(featureMatrixMinusMeanMatrix, featureMatrixMinusNewMeanMatrix)
		
		subMultipliedVarianceVector = AqwamTensorLibrary:sum(multipliedAdjustedFeatureMatrix, 1)
		
		multipliedVarianceVector = AqwamTensorLibrary:add({multipliedVarianceMatrix[classIndex]}, subMultipliedVarianceVector)
		
		newVarianceVector = AqwamTensorLibrary:divide(multipliedVarianceVector, numberOfSubData)
		
		-- End of Welford's algorithm.
		
		newStandardDeviationVector = AqwamTensorLibrary:power(newVarianceVector, 0.5)
		
		newMeanMatrix[classIndex] = newMeanVector[1]

		newStandardDeviationVector[classIndex] = newStandardDeviationVector[1]

		newPriorProbabilityVector[classIndex] = {(numberOfSubData / newTotalNumberOfDataPoint)}
		
		newNumberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return newMeanMatrix, newStandardDeviationVector, newPriorProbabilityVector, newNumberOfDataPointVector
	
end

local gaussianNaiveBayesFunctionList = {
	
	["Batch"] = batchGaussianNaiveBayes,
	
	["Sequential"] = sequentialGaussianNaiveBayes,
	
}

function GaussianNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewGaussianNaiveBayesModel = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewGaussianNaiveBayesModel, GaussianNaiveBayesModel)
	
	NewGaussianNaiveBayesModel:setName("GaussianNaiveBayes")
	
	NewGaussianNaiveBayesModel.mode = parameterDictionary.mode or defaultMode
	
	NewGaussianNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)
		
		local mode = NewGaussianNaiveBayesModel.mode
		
		local useLogProbabilities = NewGaussianNaiveBayesModel.useLogProbabilities
		
		local ModelParameters = NewGaussianNaiveBayesModel.ModelParameters or {}
		
		local meanMatrix = ModelParameters[1]
		
		local standardDeviationMatrix = ModelParameters[2]
		
		local priorProbabilityVector = ModelParameters[3]
		
		local numberOfDataPointVector = ModelParameters[4]

		if (mode == "Hybrid") then

			mode = (meanMatrix and standardDeviationMatrix and priorProbabilityVector and numberOfDataPointVector and "Sequential") or "Batch"		

		end

		local gaussianNaiveBayesFunction = gaussianNaiveBayesFunctionList[mode]

		if (not gaussianNaiveBayesFunction) then error("Unknown mode.") end
		
		if (mode == "Sequential") then
			
			local numberOfFeatures = #featureMatrix[1]
			
			local numberOfClasses = #NewGaussianNaiveBayesModel.ClassesList
			
			local zeroValue = (useLogProbabilities and math.huge) or 0
			
			local oneValue = (useLogProbabilities and 0) or 1
			
			meanMatrix = meanMatrix or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, zeroValue)
			
			standardDeviationMatrix = standardDeviationMatrix or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, zeroValue)
			
			priorProbabilityVector = priorProbabilityVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, oneValue)
			
			numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)
			
		end

		local numberOfData = #featureMatrix

		local extractedFeatureMatrixTable = NewGaussianNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)
		
		if (useLogProbabilities) then
			
			if (meanMatrix) then meanMatrix = AqwamTensorLibrary:applyFunction(math.exp, meanMatrix) end
			
			if (standardDeviationMatrix) then standardDeviationMatrix = AqwamTensorLibrary:applyFunction(math.exp, standardDeviationMatrix) end
			
			if (priorProbabilityVector) then priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.exp, priorProbabilityVector) end
			
		end
		
		meanMatrix, standardDeviationMatrix, priorProbabilityVector, numberOfDataPointVector = gaussianNaiveBayesFunction(extractedFeatureMatrixTable, numberOfData, meanMatrix, standardDeviationMatrix, priorProbabilityVector, numberOfDataPointVector)
		
		if (useLogProbabilities) then
			
			meanMatrix = AqwamTensorLibrary:applyFunction(math.log, meanMatrix)

			standardDeviationMatrix = AqwamTensorLibrary:applyFunction(math.log, standardDeviationMatrix)

			priorProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, priorProbabilityVector)
			
		end

		NewGaussianNaiveBayesModel.ModelParameters = {meanMatrix, standardDeviationMatrix, priorProbabilityVector, numberOfDataPointVector}

		local cost = NewGaussianNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewGaussianNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)
		
		local finalProbabilityVector

		local numberOfData = #featureMatrix

		local ClassesList = NewGaussianNaiveBayesModel.ClassesList

		local useLogProbabilities = NewGaussianNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewGaussianNaiveBayesModel.ModelParameters
		
		local meanMatrix = ModelParameters[1]

		local standardDeviationMatrix = ModelParameters[2]
		
		local priorProbabilityVector = ModelParameters[3]

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

		for classIndex, classValue in ipairs(ClassesList) do

			local meanVector = {meanMatrix[classIndex]}

			local standardDeviationVector = {standardDeviationMatrix[classIndex]}

			local priorProbabilityValue = {priorProbabilityVector[classIndex]}

			for i = 1, numberOfData, 1 do

				local featureVector = {featureMatrix[i]}

				posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityValue)

			end

		end

		if (returnOriginalOutput) then return posteriorProbabilityMatrix end

		return NewGaussianNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewGaussianNaiveBayesModel:setGenerateFunction(function(labelVector, noiseMatrix)
		
		if (noiseMatrix) then

			if (#labelVector ~= #noiseMatrix) then error("The label vector and the noise matrix does not contain the same number of rows.") end

		end
		
		local ClassesList = NewGaussianNaiveBayesModel.ClassesList
		
		local useLogProbabilities = NewGaussianNaiveBayesModel.useLogProbabilities
		
		local ModelParameters = NewGaussianNaiveBayesModel.ModelParameters
		
		local meanMatrix = ModelParameters[1]
		
		local standardDeviationMatrix = ModelParameters[2]
		
		local selectedMeanMatrix = {}
		
		local selectedStandardDeviationMatrix = {}
		
		for data, unwrappedLabelVector in ipairs(labelVector) do
			
			local label = unwrappedLabelVector[1]
			
			local classIndex = table.find(ClassesList, label)
			
			if (classIndex) then
				
				selectedMeanMatrix[data] = meanMatrix[classIndex]
				
				selectedStandardDeviationMatrix[data] = standardDeviationMatrix[classIndex]
				
			end
			
		end
		
		if (useLogProbabilities) then

			selectedMeanMatrix = AqwamTensorLibrary:applyFunction(math.exp, selectedMeanMatrix)

			selectedStandardDeviationMatrix = AqwamTensorLibrary:applyFunction(math.exp, selectedStandardDeviationMatrix)

		end
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selectedMeanMatrix)
		
		noiseMatrix = noiseMatrix or AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)
		
		local generatedFeatureMatrixPart1 = AqwamTensorLibrary:multiply(selectedStandardDeviationMatrix, noiseMatrix)
		
		local generatedFeatureMatrix = AqwamTensorLibrary:add(selectedMeanMatrix, generatedFeatureMatrixPart1)
		
		return generatedFeatureMatrix
		
	end)

	return NewGaussianNaiveBayesModel

end

return GaussianNaiveBayesModel
