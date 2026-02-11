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

local RecursiveLeastSquaresRegressionModel = {}

RecursiveLeastSquaresRegressionModel.__index = RecursiveLeastSquaresRegressionModel

setmetatable(RecursiveLeastSquaresRegressionModel, BaseModel)

local defaultLossFunction = "L2"

local defaultForgetFactor = 1

local defaultUseLogProbabilities = false

local defaultModelParametersInitializationMode = "Zero"

local lossFunctionList = {
	
	["L1"] = math.abs,
	
	["L2"] = function (value) return math.pow(value, 2) end
	
}

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

function RecursiveLeastSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.modelParametersInitializationMode = parameterDictionary.modelParametersInitializationMode or defaultModelParametersInitializationMode

	local NewRecursiveLeastSquaresRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewRecursiveLeastSquaresRegressionModel, RecursiveLeastSquaresRegressionModel)

	NewRecursiveLeastSquaresRegressionModel:setName("RecursiveLeastSquaresRegression")
	
	NewRecursiveLeastSquaresRegressionModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction
	
	NewRecursiveLeastSquaresRegressionModel.forgetFactor = parameterDictionary.forgetFactor or defaultForgetFactor
	
	NewRecursiveLeastSquaresRegressionModel.useLogProbabilities = NewRecursiveLeastSquaresRegressionModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	return NewRecursiveLeastSquaresRegressionModel
	
end

function RecursiveLeastSquaresRegressionModel:train(featureMatrix, labelVector)

	local numberOfData = #featureMatrix

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local lossFunction = self.lossFunction
	
	local lossFunctionToApply = lossFunctionList[lossFunction]

	if (not lossFunctionToApply) then error("Invalid loss function.") end
	
	local forgetFactor = self.forgetFactor
	
	local ModelParameters = self.ModelParameters or {}
	
	local betaVector = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfFeatures, 1})
	
	if (numberOfFeatures ~= #betaVector) then error("The number of features are not the same as the model parameters.") end
	
	local errorCovarianceMatrix = ModelParameters[2] or AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})
	
	local featureVector
	
	local responseValue
	
	local lossValue
	
	local kalmanGainVectorNumerator
	
	local transposedFeatureVector
	
	local kalmanGainVectorDenominator
	
	local kalmanGainVector
	
	local transposedKalmanGainVector
	
	local betaChangeVector
	
	local cost = 0
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		featureVector = {unwrappedFeatureVector}
		
		responseValue = AqwamTensorLibrary:dotProduct(featureVector, betaVector)[1][1]
		
		lossValue = responseValue - labelVector[dataIndex][1]
		
		kalmanGainVectorNumerator = AqwamTensorLibrary:dotProduct(featureVector, errorCovarianceMatrix) -- 1 x n
		
		transposedFeatureVector = AqwamTensorLibrary:transpose(featureVector) -- n x 1
		
		kalmanGainVectorDenominator = AqwamTensorLibrary:dotProduct(featureVector, errorCovarianceMatrix, transposedFeatureVector) -- 1 x 1
		
		kalmanGainVectorDenominator = AqwamTensorLibrary:add(forgetFactor, kalmanGainVectorDenominator)
		
		kalmanGainVector = AqwamTensorLibrary:divide(kalmanGainVectorNumerator, kalmanGainVectorDenominator) -- 1 x n
		
		transposedKalmanGainVector = AqwamTensorLibrary:transpose(kalmanGainVector)
		
		betaChangeVector = AqwamTensorLibrary:multiply(kalmanGainVector, lossValue) -- 1 x n
		
		betaChangeVector = AqwamTensorLibrary:transpose(betaChangeVector)

		betaVector = AqwamTensorLibrary:add(betaVector, betaChangeVector)
		
		errorCovarianceMatrix = AqwamTensorLibrary:subtract(errorCovarianceMatrix, AqwamTensorLibrary:dotProduct(transposedKalmanGainVector, featureVector, errorCovarianceMatrix))

		if (forgetFactor ~= 1) then errorCovarianceMatrix = AqwamTensorLibrary:divide(errorCovarianceMatrix, forgetFactor) end
		
		cost = cost + lossFunctionToApply(lossValue)
		
	end
	
	self.ModelParameters = {betaVector, errorCovarianceMatrix}
	
	cost = cost / numberOfData

	return {cost}

end

function RecursiveLeastSquaresRegressionModel:predict(featureMatrix, thresholdMatrix)

	if (thresholdMatrix) then

		if (#featureMatrix ~= #thresholdMatrix) then error("The feature matrix and the threshold matrix does not contain the same number of rows.") end

	end

	local ModelParameters = self.ModelParameters

	local betaVector

	local covarianceMatrix

	if (not ModelParameters) then

		local numberOfFeatures = #featureMatrix[1]

		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

		covarianceMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})

		self.ModelParameters = {betaVector, covarianceMatrix}

	else

		betaVector = ModelParameters[1]

		covarianceMatrix = ModelParameters[2]

	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)

	if (not thresholdMatrix) then return predictedVector end

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local predictedVarianceVectorPart1 = AqwamTensorLibrary:dotProduct(featureMatrix, covarianceMatrix)

	local predictedVarianceVectorPart2 = AqwamTensorLibrary:dotProduct(predictedVarianceVectorPart1, transposedFeatureMatrix)

	local predictedVarianceVector = {}

	for i, predictedVarianceTable in ipairs(predictedVarianceVectorPart2) do

		predictedVarianceVector[i] = {predictedVarianceTable[i]}

	end

	local predictedStandardDeviationVector = AqwamTensorLibrary:applyFunction(math.sqrt, predictedVarianceVector)

	local probabilityMatrix = calculateGaussianProbability(self.useLogProbabilities, thresholdMatrix, predictedVector, predictedStandardDeviationVector)

	return predictedVector, probabilityMatrix 

end

return RecursiveLeastSquaresRegressionModel
