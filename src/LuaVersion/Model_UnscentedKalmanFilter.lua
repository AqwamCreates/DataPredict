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

local UnscentedKalmanFilterModel = {}

UnscentedKalmanFilterModel.__index = UnscentedKalmanFilterModel

setmetatable(UnscentedKalmanFilterModel, BaseModel)

local defaultAlpha = 1e-3

local defaultBeta = 2

local defaultKappa = 0

local defaultNoiseValue = 1 -- Do not use very small value for this. It will cause the Mahalanobis distance to have very large values.

local defaultLossFunction = "L2"

local function defaultStateTransitionFunction(stateMatrix, deltaTime)
	
	return stateMatrix
	
end

local function defaultObservationFunction(stateMatrix)

	return stateMatrix
	
end

function UnscentedKalmanFilterModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewUKFModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewUKFModel, UnscentedKalmanFilterModel)
	
	NewUKFModel:setName("UnscentedKalmanFilter")
	
	NewUKFModel.alpha = parameterDictionary.alpha or defaultAlpha
	
	NewUKFModel.beta = parameterDictionary.beta or defaultBeta
	
	NewUKFModel.kappa = parameterDictionary.kappa or defaultKappa

	NewUKFModel.noiseValue = parameterDictionary.noiseValue or defaultNoiseValue
	
	NewUKFModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction

	NewUKFModel.stateTransitionFunction = parameterDictionary.stateTransitionFunction or defaultStateTransitionFunction
	
	NewUKFModel.observationFunction = parameterDictionary.observationFunction or defaultObservationFunction

	NewUKFModel.processNoiseCovarianceMatrix = parameterDictionary.processNoiseCovarianceMatrix
	
	NewUKFModel.observationNoiseCovarianceMatrix = parameterDictionary.observationNoiseCovarianceMatrix

	return NewUKFModel
end

local function generateSigmaPoints(meanStateMatrix, covarianceMatrix, alpha, kappa)

	local numberOfStates = #meanStateMatrix
	
	local lambdaValue = math.pow(alpha, 2) * (numberOfStates + kappa) - numberOfStates

	local scaledCovarianceMatrix = AqwamTensorLibrary:multiply(covarianceMatrix, (numberOfStates + lambdaValue))

	local squareRootCovarianceMatrix = AqwamTensorLibrary:applyFunction(math.sqrt, scaledCovarianceMatrix)

	local sigmaPointMatrixArray = {}
	
	table.insert(sigmaPointMatrixArray, meanStateMatrix)
	
	local columnVector 
	
	local sigmaPlus
	
	local sigmaMinus

	for i = 1, numberOfStates, 1 do

		columnVector = AqwamTensorLibrary:extract(squareRootCovarianceMatrix, {1, i}, {numberOfStates, i})

		sigmaPlus = AqwamTensorLibrary:add(meanStateMatrix, columnVector)
		
		sigmaMinus = AqwamTensorLibrary:subtract(meanStateMatrix, columnVector)

		table.insert(sigmaPointMatrixArray, sigmaPlus)
		
		table.insert(sigmaPointMatrixArray, sigmaMinus)
		
	end

	return sigmaPointMatrixArray, lambdaValue
	
end

local function calculateWeightedMeanMatrix(matrixArray, weightArray)
	
	local numberOfMatrices = #matrixArray
	
	local weightedMeanMatrix = AqwamTensorLibrary:multiply(matrixArray[1], weightArray[1])
	
	local meanMatrix

	for i = 2, numberOfMatrices do
		
		meanMatrix = AqwamTensorLibrary:multiply(matrixArray[i], weightArray[i])
		
		weightedMeanMatrix = AqwamTensorLibrary:add(weightedMeanMatrix, meanMatrix)
		
	end

	return weightedMeanMatrix
	
end

local function calculateWeightedCovariance(matrixArray, weightArray, meanMatrix)
	
	local weightedCovarianceMatrix
	
	local covarianceMatrixPart1
	
	local transposedCovarianceMatrixPart1
	
	local covarianceMatrixPart2
	
	local covarianceMatrix
	
	for i, matrix in ipairs(matrixArray) do
		
		covarianceMatrixPart1 = AqwamTensorLibrary:subtract(matrix, meanMatrix)
		
		transposedCovarianceMatrixPart1 = AqwamTensorLibrary:transpose(covarianceMatrixPart1)
		
		covarianceMatrixPart2 = AqwamTensorLibrary:dotProduct(covarianceMatrixPart1, transposedCovarianceMatrixPart1)
		
		covarianceMatrix = AqwamTensorLibrary:multiply(weightArray[i], covarianceMatrixPart2)
		
		if (i == 1) then
			
			weightedCovarianceMatrix = covarianceMatrix
			
		else
			
			weightedCovarianceMatrix = AqwamTensorLibrary:add(weightedCovarianceMatrix, covarianceMatrix)
			
		end
		
	end

	return weightedCovarianceMatrix
	
end

local function calculateCrossVariance(stateSigmaMatrixArray, meanStateMatrix, observationSigmaArray, meanObsMatrix, weightArray)
	
	local numberOfSigmaPoints = #stateSigmaMatrixArray

	local crossCovarianceMatrix

	local stateDeviationMatrix

	local observationDeviationMatrix

	local transposedObservationDeviationMatrix

	local weightedCrossMatrixPart1 

	local weightedCrossMatrix

	for i = 1, numberOfSigmaPoints do

		stateDeviationMatrix = AqwamTensorLibrary:subtract(stateSigmaMatrixArray[i], meanStateMatrix)

		observationDeviationMatrix = AqwamTensorLibrary:subtract(observationSigmaArray[i], meanObsMatrix)

		transposedObservationDeviationMatrix = AqwamTensorLibrary:transpose(observationDeviationMatrix)

		weightedCrossMatrixPart1 = AqwamTensorLibrary:dotProduct(stateDeviationMatrix, transposedObservationDeviationMatrix)

		weightedCrossMatrix = AqwamTensorLibrary:multiply(weightArray[i], weightedCrossMatrixPart1)

		if (i == 1) then

			crossCovarianceMatrix = weightedCrossMatrix

		else

			crossCovarianceMatrix = AqwamTensorLibrary:add(crossCovarianceMatrix, weightedCrossMatrix)

		end

	end

	return crossCovarianceMatrix
	
end

function UnscentedKalmanFilterModel:train(previousStateMatrix, currentStateMatrix)
	
	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end

	local numberOfStates = #previousStateMatrix[1]

	if (numberOfStates ~= #currentStateMatrix[1]) then error("The number of states in the previous state vector is not equal to the number of states in the current state vector.") end

	local numberOfStatesDimensionSizeArray = {numberOfStates, numberOfStates}
	
	previousStateMatrix = AqwamTensorLibrary:transpose(previousStateMatrix)

	currentStateMatrix = AqwamTensorLibrary:transpose(currentStateMatrix)
	
	local alpha = self.alpha

	local beta = self.beta

	local kappa = self.kappa

	local noiseValue = self.noiseValue

	local lossFunction = self.lossFunction
	
	local stateTransitionFunction = self.stateTransitionFunction
	
	local observationFunction = self.observationFunction
	
	local processNoiseCovarianceMatrix = self.processNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)
	
	local observationNoiseCovarianceMatrix = self.observationNoiseCovarianceMatrix or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)

	local ModelParameters = self.ModelParameters or {}
	
	local priorMeanStateMatrix = ModelParameters[1] or previousStateMatrix
	
	local priorCovarianceMatrix = ModelParameters[2] or AqwamTensorLibrary:createIdentityTensor(numberOfStatesDimensionSizeArray, noiseValue)

	local sigmaPointMatrixArray, lambdaValue = generateSigmaPoints(priorMeanStateMatrix, priorCovarianceMatrix, alpha, kappa)
	
	local numberOfSigmaPoints = #sigmaPointMatrixArray

	local weightMeanArray = {}
	
	local weightCovarianceArray = {}

	weightMeanArray[1] = lambdaValue / (numberOfStates + lambdaValue)
	
	weightCovarianceArray[1] = weightMeanArray[1] + (1 - alpha^2 + beta)

	for i = 2, numberOfSigmaPoints, 1 do
		
		weightMeanArray[i] = 1 / (2 * (numberOfStates + lambdaValue))
		
		weightCovarianceArray[i] = weightMeanArray[i]
		
	end

	local predictedSigmaPointMatrixArray = {}
	
	for i, sigmaPointMatrix in ipairs(sigmaPointMatrixArray) do
		
		predictedSigmaPointMatrixArray[i] = stateTransitionFunction(sigmaPointMatrix)
		
	end

	local predictedMeanStateMatrix = calculateWeightedMeanMatrix(predictedSigmaPointMatrixArray, weightMeanArray)
	
	local predictedCovarianceMatrix = calculateWeightedCovariance(predictedSigmaPointMatrixArray, weightCovarianceArray, predictedMeanStateMatrix)
	
	predictedCovarianceMatrix = AqwamTensorLibrary:add(predictedCovarianceMatrix, processNoiseCovarianceMatrix)

	local predictedObservationSigmaPointsArray = {}
	
	for i, predictedSigmaPointMatrix in ipairs(predictedSigmaPointMatrixArray) do
		
		predictedObservationSigmaPointsArray[i] = observationFunction(predictedSigmaPointMatrix)
		
	end

	local predictedMeanObservationMatrix = calculateWeightedMeanMatrix(predictedObservationSigmaPointsArray, weightMeanArray)
	
	local innovationCovarianceMatrix = calculateWeightedCovariance(predictedObservationSigmaPointsArray, weightCovarianceArray, predictedMeanObservationMatrix)
	
	innovationCovarianceMatrix = AqwamTensorLibrary:add(innovationCovarianceMatrix, observationNoiseCovarianceMatrix)

	local crossCovarianceMatrix = calculateCrossVariance(predictedSigmaPointMatrixArray, predictedMeanStateMatrix, predictedObservationSigmaPointsArray, predictedMeanObservationMatrix, weightCovarianceArray)

	local inverseInnovationCovarianceMatrix = AqwamTensorLibrary:inverse(innovationCovarianceMatrix)
	
	local kalmanGainMatrix = AqwamTensorLibrary:dotProduct(crossCovarianceMatrix, inverseInnovationCovarianceMatrix)

	local innovationMatrix = AqwamTensorLibrary:subtract(currentStateMatrix, predictedMeanObservationMatrix)
	
	local posteriorStateMatrixPart1 = AqwamTensorLibrary:dotProduct(kalmanGainMatrix, innovationMatrix)
	
	local posteriorMeanStateMatrix = AqwamTensorLibrary:add(posteriorStateMatrixPart1, predictedMeanStateMatrix)
	
	local transposedKalmanGainMatrix = AqwamTensorLibrary:transpose(kalmanGainMatrix)
	
	local posteriorCovarianceMatrixPart1 = AqwamTensorLibrary:dotProduct(kalmanGainMatrix, inverseInnovationCovarianceMatrix, transposedKalmanGainMatrix)

	local posteriorCovarianceMatrix = AqwamTensorLibrary:subtract(priorCovarianceMatrix, posteriorCovarianceMatrixPart1)
	
	local meanCorrectionMatrix = AqwamTensorLibrary:mean(posteriorStateMatrixPart1, 2)

	self.ModelParameters = {posteriorMeanStateMatrix, posteriorCovarianceMatrix, meanCorrectionMatrix}

	-- Returning this as a cost like other models.
	
	local lossMatrix = innovationMatrix

	if (lossFunction == "L1") then

		lossMatrix = AqwamTensorLibrary:applyFunction(math.abs, lossMatrix)

	elseif (lossFunction == "L2") then

		lossMatrix = AqwamTensorLibrary:power(lossMatrix, 2)

	elseif (lossFunction  == "Mahalanobis") then

		local transposedInnovationMatrix = AqwamTensorLibrary:transpose(innovationMatrix)

		lossMatrix = AqwamTensorLibrary:dotProduct(transposedInnovationMatrix, inverseInnovationCovarianceMatrix, innovationMatrix)

	else

		error("Invalid loss function.")

	end

	local cost = AqwamTensorLibrary:sum(lossMatrix)

	cost = cost / numberOfData

	return {cost}
	
end

function UnscentedKalmanFilterModel:predict(stateMatrix)
	
	local ModelParameters = self.ModelParameters or {}

	local meanCorrectionMatrix = ModelParameters[3]

	stateMatrix = AqwamTensorLibrary:transpose(stateMatrix)

	local nextStateMatrix = self.stateTransitionFunction(stateMatrix)

	if (meanCorrectionMatrix) then nextStateMatrix = AqwamTensorLibrary:add(nextStateMatrix, meanCorrectionMatrix) end

	nextStateMatrix = AqwamTensorLibrary:transpose(nextStateMatrix)

	return nextStateMatrix
	
end

return UnscentedKalmanFilterModel
