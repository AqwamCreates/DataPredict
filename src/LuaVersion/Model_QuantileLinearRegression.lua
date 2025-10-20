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

local BaseModel = require("Model_BaseModel")

local zTableFunction = require("Core_ZTableFunction")

QuantileLinearRegressionModel = {}

QuantileLinearRegressionModel.__index = QuantileLinearRegressionModel

setmetatable(QuantileLinearRegressionModel, BaseModel)

local defaultPriorPrecision = 1.0 -- alpha

local defaultLikelihoodPrecision = 1.0 -- beta

function QuantileLinearRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewQuantileLinearRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewQuantileLinearRegressionModel, QuantileLinearRegressionModel)

	NewQuantileLinearRegressionModel:setName("QuantileLinearRegression")

	NewQuantileLinearRegressionModel.priorPrecision = parameterDictionary.priorPrecision or defaultPriorPrecision

	NewQuantileLinearRegressionModel.likelihoodPrecision = parameterDictionary.likelihoodPrecision or defaultLikelihoodPrecision

	return NewQuantileLinearRegressionModel
	
end

function QuantileLinearRegressionModel:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end

	local priorPrecision = self.priorPrecision

	local likelihoodPrecision = self.likelihoodPrecision

	local numberOfFeatures = #featureMatrix[1]
	
	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	local alphaI = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})

	alphaI = AqwamTensorLibrary:multiply(alphaI, priorPrecision)

	local betaXTX = AqwamTensorLibrary:multiply(dotProductFeatureMatrix, likelihoodPrecision)

	local S_N_inv = AqwamTensorLibrary:add(alphaI, betaXTX)

	local posteriorCovariance = AqwamTensorLibrary:inverse(S_N_inv)
	
	if (not posteriorCovariance) then error("Could not invert matrix for posterior.") end
	
	local dotProductFeatureMatrixLabelVector = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, labelVector)

	local temporaryMatrix = AqwamTensorLibrary:multiply(posteriorCovariance, dotProductFeatureMatrixLabelVector)

	local posteriorMeanMatrix = AqwamTensorLibrary:multiply(temporaryMatrix, likelihoodPrecision)

	self.ModelParameters = {posteriorMeanMatrix, posteriorCovariance}

end

function QuantileLinearRegressionModel:predict(featureMatrix, quantileVector)
	
	if (quantileVector) then
		
		if (#featureMatrix ~= #quantileVector) then error("The feature matrix and the quantile vector does not contain the same number of rows.") end
		
	end

	local ModelParameters = self.ModelParameters
	
	local posteriorMeanMatrix
	
	local posteriorCovarianceMatrix

	if (not ModelParameters) then
		
		local dimensionSizeArray = {#featureMatrix[1], 1}

		posteriorMeanMatrix = self:initializeMatrixBasedOnMode(dimensionSizeArray)
		
		posteriorCovarianceMatrix = AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)

		self.ModelParameters = {posteriorMeanMatrix, posteriorCovarianceMatrix}
		
	else
		
		posteriorMeanMatrix = ModelParameters[1]

		posteriorCovarianceMatrix = ModelParameters[2]

	end
	
	local predictedMeanVector = AqwamTensorLibrary:dotProduct(featureMatrix, posteriorMeanMatrix)

	if (not quantileVector) then return predictedMeanVector end
	
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
	
	local predictedQuantileVector = {}
	
	local zValue
	
	for i, probabilityTable in ipairs(quantileVector) do
		
		zValue = zTableFunction:getStandardNormalInverseCumulativeDistributionFunction(probabilityTable[1])
		
		predictedQuantileVector[i] = {predictedMeanVector[i][1] + zValue * predictedStandardDeviationVector[i][1]}
		
	end

	return predictedMeanVector, predictedQuantileVector 
	
end

return QuantileLinearRegressionModel
