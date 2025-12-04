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

local ZTableFunction = require(script.Parent.Parent.Cores.ZTableFunction)

local BayesianQuantileLinearRegressionModel = {}

BayesianQuantileLinearRegressionModel.__index = BayesianQuantileLinearRegressionModel

setmetatable(BayesianQuantileLinearRegressionModel, BaseModel)

local defaultPriorPrecision = 1.0 -- alpha

local defaultLikelihoodPrecision = 1.0 -- beta

function BayesianQuantileLinearRegressionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewBayesianQuantileLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewBayesianQuantileLinearRegressionModel, BayesianQuantileLinearRegressionModel)

	NewBayesianQuantileLinearRegressionModel:setName("BayesianQuantileLinearRegression")

	NewBayesianQuantileLinearRegressionModel.priorPrecision = parameterDictionary.priorPrecision or defaultPriorPrecision

	NewBayesianQuantileLinearRegressionModel.likelihoodPrecision = parameterDictionary.likelihoodPrecision or defaultLikelihoodPrecision

	return NewBayesianQuantileLinearRegressionModel

end

function BayesianQuantileLinearRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local priorPrecision = self.priorPrecision

	local likelihoodPrecision = self.likelihoodPrecision

	local numberOfFeatures = #featureMatrix[1]

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	local priorPrecisionIdentityMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})

	priorPrecisionIdentityMatrix = AqwamTensorLibrary:multiply(priorPrecisionIdentityMatrix, priorPrecision)

	local scaledDotProductFeatureMatrix = AqwamTensorLibrary:multiply(dotProductFeatureMatrix, likelihoodPrecision)

	local inverseSNMatrix = AqwamTensorLibrary:add(priorPrecisionIdentityMatrix, scaledDotProductFeatureMatrix)

	local posteriorCovarianceMatrix = AqwamTensorLibrary:inverse(inverseSNMatrix)

	if (not posteriorCovarianceMatrix) then error("Could not invert matrix for posterior.") end

	local dotProductFeatureMatrixLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)

	local posteriorMeanVectorPart1 = AqwamTensorLibrary:dotProduct(posteriorCovarianceMatrix, dotProductFeatureMatrixLabelVector)

	local posteriorMeanVector = AqwamTensorLibrary:multiply(posteriorMeanVectorPart1, likelihoodPrecision)

	self.ModelParameters = {posteriorMeanVector, posteriorCovarianceMatrix}

end

function BayesianQuantileLinearRegressionModel:predict(featureMatrix, quantileMatrix)

	local numberOfData = #featureMatrix

	if (quantileMatrix) then

		if (numberOfData ~= #quantileMatrix) then error("The feature matrix and the quantile matrix does not contain the same number of rows.") end

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

	if (not quantileMatrix) then return predictedMeanVector end

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

	local numberOfQuantiles = #quantileMatrix[1]

	local predictedQuantileMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfQuantiles}, 0)
	
	local unwrappedPredictedQuantileVector
	
	local predictedMeanValue
	
	local predictedStandardDeviationValue

	local zValue

	for i, unwrappedQuantileVector in ipairs(quantileMatrix) do
		
		unwrappedPredictedQuantileVector = predictedQuantileMatrix[i]
		
		predictedMeanValue = predictedMeanVector[i][1]
		
		predictedStandardDeviationValue = predictedStandardDeviationVector[i][1]
		
		for j, quantileValue in ipairs(unwrappedQuantileVector) do
			
			zValue = ZTableFunction:calculateStandardNormalInverseCumulativeDistributionValue(quantileValue)
			
			unwrappedPredictedQuantileVector[j] = predictedMeanValue + (zValue * predictedStandardDeviationValue)
			
		end
		
		predictedQuantileMatrix[i] = unwrappedPredictedQuantileVector

	end

	return predictedMeanVector, predictedQuantileMatrix 

end

return BayesianQuantileLinearRegressionModel
