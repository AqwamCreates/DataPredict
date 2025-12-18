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

local BayesianLinearRegressionModel = {}

BayesianLinearRegressionModel.__index = BayesianLinearRegressionModel

setmetatable(BayesianLinearRegressionModel, BaseModel)

local defaultPriorPrecision = 1.0 -- alpha

local defaultLikelihoodPrecision = 1.0 -- beta

local defaultUseLogProbabilities = false

local function calculateGaussianProbability(useLogProbabilities, thresholdVector, meanVector, standardDeviationVector)

	local gaussianProbability = (useLogProbabilities and 0) or 1

	local exponentStep1Vector = AqwamTensorLibrary:subtract(thresholdVector, meanVector)

	local exponentStep2Vector = AqwamTensorLibrary:power(exponentStep1Vector, 2)

	local exponentPart3Vector = AqwamTensorLibrary:power(standardDeviationVector, 2)

	local exponentStep4Vector = AqwamTensorLibrary:divide(exponentStep2Vector, exponentPart3Vector)

	local exponentStep5Vector = AqwamTensorLibrary:multiply(-0.5, exponentStep4Vector)

	local exponentWithTermsVector = AqwamTensorLibrary:applyFunction(math.exp, exponentStep5Vector)

	local divisorVector = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local gaussianProbabilityVector = AqwamTensorLibrary:divide(exponentWithTermsVector, divisorVector)

	if (useLogProbabilities) then gaussianProbabilityVector = AqwamTensorLibrary:applyFunction(math.log, gaussianProbabilityVector) end

	return gaussianProbabilityVector

end

function BayesianLinearRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewBayesianLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewBayesianLinearRegressionModel, BayesianLinearRegressionModel)

	NewBayesianLinearRegressionModel:setName("BayesianLinearRegression")

	NewBayesianLinearRegressionModel.priorPrecision = parameterDictionary.priorPrecision or defaultPriorPrecision

	NewBayesianLinearRegressionModel.likelihoodPrecision = parameterDictionary.likelihoodPrecision or defaultLikelihoodPrecision
	
	NewBayesianLinearRegressionModel.useLogProbabilities = NewBayesianLinearRegressionModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	return NewBayesianLinearRegressionModel
	
end

function BayesianLinearRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local priorPrecision = self.priorPrecision

	local likelihoodPrecision = self.likelihoodPrecision

	local ModelParameters = self.ModelParameters or {}

	local oldPosteriorMeanVector = ModelParameters[1]

	local oldInversePosteriorCovarianceMatrix = ModelParameters[3]

	local oldMultipliedDotProductFeatureMatrixLabelVector = ModelParameters[4]

	local numberOfFeatures = #featureMatrix[1]

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	local priorPrecisionIdentityMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures}, priorPrecision)

	local scaledDotProductFeatureMatrix = AqwamTensorLibrary:multiply(dotProductFeatureMatrix, likelihoodPrecision)

	local inversePosteriorCovarianceMatrix = AqwamTensorLibrary:add(priorPrecisionIdentityMatrix, scaledDotProductFeatureMatrix)

	if (oldInversePosteriorCovarianceMatrix) then

		inversePosteriorCovarianceMatrix = AqwamTensorLibrary:add(inversePosteriorCovarianceMatrix, oldInversePosteriorCovarianceMatrix)

	end

	local posteriorCovarianceMatrix = AqwamTensorLibrary:inverse(inversePosteriorCovarianceMatrix)

	if (not posteriorCovarianceMatrix) then error("Could not invert matrix for posterior covariance matrix.") end

	local dotProductFeatureMatrixLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)

	local multipliedDotProductFeatureMatrixLabelVector = AqwamTensorLibrary:multiply(likelihoodPrecision, dotProductFeatureMatrixLabelVector)

	if (oldMultipliedDotProductFeatureMatrixLabelVector) then

		multipliedDotProductFeatureMatrixLabelVector = AqwamTensorLibrary:add(oldMultipliedDotProductFeatureMatrixLabelVector, multipliedDotProductFeatureMatrixLabelVector)

	end

	local posteriorMeanVector = AqwamTensorLibrary:dotProduct(posteriorCovarianceMatrix, multipliedDotProductFeatureMatrixLabelVector)

	self.ModelParameters = {posteriorMeanVector, posteriorCovarianceMatrix, inversePosteriorCovarianceMatrix, multipliedDotProductFeatureMatrixLabelVector}

end

function BayesianLinearRegressionModel:predict(featureMatrix, thresholdMatrix)
	
	if (thresholdMatrix) then
		
		if (#featureMatrix ~= #thresholdMatrix) then error("The feature matrix and the threshold matrix does not contain the same number of rows.") end
		
	end

	local ModelParameters = self.ModelParameters
	
	local posteriorMeanVector
	
	local posteriorCovarianceMatrix

	if (not ModelParameters) then
		
		local dimensionSizeArray = {#featureMatrix[1], 1}

		posteriorMeanVector = self:initializeMatrixBasedOnMode(dimensionSizeArray)
		
		posteriorCovarianceMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)

		self.ModelParameters = {posteriorMeanVector, posteriorCovarianceMatrix}
		
	else
		
		posteriorMeanVector = ModelParameters[1]

		posteriorCovarianceMatrix = ModelParameters[2]

	end
	
	local predictedMeanVector = AqwamTensorLibrary:dotProduct(featureMatrix, posteriorMeanVector)

	if (not thresholdMatrix) then return predictedMeanVector end
	
	local likelihoodPrecision = self.likelihoodPrecision
	
	local inverseLikelihoodPrecision = 1 / likelihoodPrecision
	
	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
	
	local predictedVarianceVectorPart1 = AqwamTensorLibrary:dotProduct(featureMatrix, posteriorCovarianceMatrix)
	
	local predictedVarianceVectorPart2 = AqwamTensorLibrary:dotProduct(predictedVarianceVectorPart1, transposedFeatureMatrix)
	
	local predictedVarianceVector = {}
	
	for i, predictedVarianceTable in ipairs(predictedVarianceVectorPart2) do
		
		predictedVarianceVector[i] = {predictedVarianceTable[i] + inverseLikelihoodPrecision}
		
	end
	
	local predictedStandardDeviationVector = AqwamTensorLibrary:applyFunction(math.sqrt, predictedVarianceVector)
	
	local probabilityMatrix = calculateGaussianProbability(self.useLogProbabilities, thresholdMatrix, predictedMeanVector, predictedStandardDeviationVector)

	return predictedMeanVector, probabilityMatrix 
	
end

return BayesianLinearRegressionModel
