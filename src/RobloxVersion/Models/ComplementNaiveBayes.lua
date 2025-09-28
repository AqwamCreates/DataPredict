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

ComplementNaiveBayesModel = {}

ComplementNaiveBayesModel.__index = ComplementNaiveBayesModel

setmetatable(ComplementNaiveBayesModel, NaiveBayesBaseModel)

local defaultMode = "Hybrid"

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

local function batchComplementNaiveBayes(extractedFeatureMatrixTable, numberOfData)
	
	local complementFeatureProbabilityMatrix = {}

	local priorProbabilityVector = {}

	local numberOfDataPointVector = {}

	local extractedFeatureMatrix

	local extractedComplementFeatureMatrix

	local sumExtractedComplementFeatureVector

	local totalSumExtractedComplementFeatureVector

	local complementFeatureProbabilityVector

	local numberOfSubData

	local numberOfComplementSubData

	local totalNumberOfComplementSubData

	for classIndex, extractedFeatureMatrix in ipairs(extractedFeatureMatrixTable) do

		numberOfSubData = #extractedFeatureMatrix

		totalSumExtractedComplementFeatureVector = nil

		totalNumberOfComplementSubData = 0
		
		for complementClassIndex, extractedComplementFeatureMatrix in ipairs(extractedFeatureMatrixTable) do
			
			if (complementClassIndex ~= classIndex) then

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

		priorProbabilityVector[classIndex] = {(numberOfSubData / numberOfData)}
		
		numberOfDataPointVector[classIndex] = {numberOfSubData}
		
	end
	
	return complementFeatureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector
	
end

local function sequentialComplementNaiveBayes(extractedFeatureMatrixTable, numberOfData)
	
	
	
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

		local featureProbabilityMatrix = ModelParameters[1]

		local priorProbabilityVector = ModelParameters[2]

		local numberOfDataPointVector = ModelParameters[3]

		if (mode == "Hybrid") then

			mode = (featureProbabilityMatrix and priorProbabilityVector and numberOfDataPointVector and "Sequential") or "Batch"		

		end
		
		local complementNaiveBayesFunction = complementNaiveBayesFunctionList[mode]

		if (not complementNaiveBayesFunction) then error("Unknown mode.") end

		local numberOfData = #featureMatrix
		
		local extractedFeatureMatrixTable = NewComplementNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)
		
		featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector = complementNaiveBayesFunction(extractedFeatureMatrixTable, numberOfData, featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector)

		NewComplementNaiveBayesModel.ModelParameters = {featureProbabilityMatrix, priorProbabilityVector, numberOfDataPointVector}

		local cost = NewComplementNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewComplementNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)
		
		local finalProbabilityVector

		local numberOfData = #featureMatrix

		local ClassesList = NewComplementNaiveBayesModel.ClassesList

		local useLogProbabilities = NewComplementNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewComplementNaiveBayesModel.ModelParameters
		
		local complementFeatureProbabilityMatrix = ModelParameters[1]
		
		local priorProbabilityVector = ModelParameters[2]

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

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

	return NewComplementNaiveBayesModel

end

return ComplementNaiveBayesModel
