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

local NaiveBayesBaseModel = require(script.Parent.NaiveBayesBaseModel)

GaussianNaiveBayesModel = {}

GaussianNaiveBayesModel.__index = GaussianNaiveBayesModel

setmetatable(GaussianNaiveBayesModel, NaiveBayesBaseModel)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

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

	local cost

	local featureVector

	local meanVector

	local standardDeviationVector

	local priorProbabilityVector

	local posteriorProbability

	local probability

	local classIndex

	local label

	local numberOfData = #labelVector

	local useLogProbabilities = self.useLogProbabilities
	
	local ModelParameters = self.ModelParameters
	
	local ClassesList = self.ClassesList

	local posteriorProbabilityVector = AqwamTensorLibrary:createTensor({numberOfData, #labelVector[1]})
	
	for data, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		featureVector = {unwrappedFeatureVector}

		label = labelVector[data][1]

		classIndex = table.find(ClassesList, label)

		meanVector = {ModelParameters[1][classIndex]}

		standardDeviationVector = {ModelParameters[2][classIndex]}

		priorProbabilityVector = {ModelParameters[3][classIndex]}

		posteriorProbabilityVector[data][1] = calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityVector)
		
	end

	cost = self:logLoss(labelVector, posteriorProbabilityVector)

	return cost

end

function GaussianNaiveBayesModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewGaussianNaiveBayesModel = NaiveBayesBaseModel.new(parameterDictionary)

	setmetatable(NewGaussianNaiveBayesModel, GaussianNaiveBayesModel)
	
	NewGaussianNaiveBayesModel:setName("GaussianNaiveBayes")
	
	NewGaussianNaiveBayesModel:setTrainFunction(function(featureMatrix, labelVector)
		
		local cost

		local extractedFeatureMatrix

		local meanVector

		local standardDeviationVector

		local numberOfSubData

		local ModelParameters = NewGaussianNaiveBayesModel.ModelParameters

		local ClassesList = NewGaussianNaiveBayesModel.ClassesList

		local numberOfClasses = #ClassesList

		local numberOfData = #featureMatrix

		local numberOfFeatures = #featureMatrix[1]

		local extractedFeatureMatricesTable = NewGaussianNaiveBayesModel:separateFeatureMatrixByClass(featureMatrix, labelVector)

		local meanMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

		local standardDeviationMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)

		local priorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfClasses, 1})

		for classIndex, classValue in ipairs(ClassesList) do

			extractedFeatureMatrix = extractedFeatureMatricesTable[classIndex]

			numberOfSubData = #extractedFeatureMatrix

			standardDeviationVector, _, meanVector = AqwamTensorLibrary:standardDeviation(extractedFeatureMatrix, 1)

			meanMatrix[classIndex] = meanVector[1]

			standardDeviationMatrix[classIndex] = standardDeviationVector[1]

			priorProbabilityMatrix[classIndex] = {(numberOfSubData / numberOfData)}

		end

		if (ModelParameters) then

			meanMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[1], meanMatrix), 2) 

			standardDeviationMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[2], standardDeviationMatrix), 2) 

			priorProbabilityMatrix = AqwamTensorLibrary:divide(AqwamTensorLibrary:add(ModelParameters[3], priorProbabilityMatrix), 2) 

		end

		NewGaussianNaiveBayesModel.ModelParameters = {meanMatrix, standardDeviationMatrix, priorProbabilityMatrix}

		cost = NewGaussianNaiveBayesModel:calculateCost(featureMatrix, labelVector)

		return {cost}
		
	end)
	
	NewGaussianNaiveBayesModel:setPredictFunction(function(featureMatrix, returnOriginalOutput)
		
		local finalProbabilityVector

		local numberOfData = #featureMatrix

		local ClassesList = NewGaussianNaiveBayesModel.ClassesList

		local useLogProbabilities = NewGaussianNaiveBayesModel.useLogProbabilities

		local ModelParameters = NewGaussianNaiveBayesModel.ModelParameters

		local posteriorProbabilityMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

		for classIndex, classValue in ipairs(ClassesList) do

			local meanVector = {ModelParameters[1][classIndex]}

			local standardDeviationVector = {ModelParameters[2][classIndex]}

			local priorProbabilityVector = {ModelParameters[3][classIndex]}

			for i = 1, numberOfData, 1 do

				local featureVector = {featureMatrix[i]}

				posteriorProbabilityMatrix[i][classIndex] = calculatePosteriorProbability(useLogProbabilities, featureVector, meanVector, standardDeviationVector, priorProbabilityVector)

			end

		end

		if (returnOriginalOutput) then return posteriorProbabilityMatrix end

		return NewGaussianNaiveBayesModel:getLabelFromOutputMatrix(posteriorProbabilityMatrix)
		
	end)
	
	NewGaussianNaiveBayesModel:setGenerateFunction(function(labelVector)
		
		local ClassesList = NewGaussianNaiveBayesModel.ClassesList
		
		local useLogProbabilities = NewGaussianNaiveBayesModel.useLogProbabilities
		
		local ModelParameters = NewGaussianNaiveBayesModel.ModelParameters
		
		local selectedMeanMatrix = {}
		
		local selectedStandardDeviationMatrix = {}
		
		for data, unwrappedLabelVector in ipairs(labelVector) do
			
			local label = unwrappedLabelVector[1]
			
			local classIndex = table.find(ClassesList, label)
			
			if (classIndex) then
				
				selectedMeanMatrix[data] = ModelParameters[1][classIndex]
				
				selectedStandardDeviationMatrix[data] = ModelParameters[2][classIndex]
				
			end
			
		end
		
		if (useLogProbabilities) then

			selectedMeanMatrix = AqwamTensorLibrary:applyFunction(math.exp, selectedMeanMatrix)

			selectedStandardDeviationMatrix = AqwamTensorLibrary:applyFunction(math.exp, selectedStandardDeviationMatrix)

		end
		
		local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(selectedMeanMatrix)
		
		local noiseMatrix = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)
		
		local generatedFeatureMatrixPart1 = AqwamTensorLibrary:multiply(selectedStandardDeviationMatrix, noiseMatrix)
		
		local generatedFeatureMatrix = AqwamTensorLibrary:add(selectedMeanMatrix, generatedFeatureMatrixPart1)
		
		return generatedFeatureMatrix
		
	end)

	return NewGaussianNaiveBayesModel

end

return GaussianNaiveBayesModel
