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

local GradientMethodBaseModel = require("Model_GradientMethodBaseModel")

local distanceFunctionDictionary = require("Core_DistanceFunctionDictionary")

local SupportVectorMachineIterativeReweightedLeastSquaresVariantModel = {}

SupportVectorMachineIterativeReweightedLeastSquaresVariantModel.__index = SupportVectorMachineIterativeReweightedLeastSquaresVariantModel

setmetatable(SupportVectorMachineIterativeReweightedLeastSquaresVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 1

local defaultCValue = 1

local defaultKernelFunction = "Linear"

local defaultGamma = 1

local defaultDegree = 3

local defaultSigma = 1

local defaultR = 0

local function hingeFunction(value)
	
	return math.max(0, value)
	
end

local function misclassificationMaskFunction(value)
	
	return (value < 1) and 1 or 0
	
end

local seperatorFunction = function (x) 

	return ((x > 0) and 1) or ((x < 0) and -1) or 0

end

local function createDistanceMatrix(distanceFunction, matrix1, matrix2)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData1, numberOfData2})

	local distanceFunctionToApply = distanceFunctionDictionary[distanceFunction]

	for i = 1, numberOfData1, 1 do

		for j = 1, numberOfData2, 1 do

			distanceMatrix[i][j] = distanceFunctionToApply({matrix1[i]}, {matrix2[j]})

		end

	end

	return distanceMatrix

end

local mappingList = {

	["Linear"] = function(featureMatrix)

		return featureMatrix

	end,

	["Polynomial"] = function(featureMatrix, kernelParameters)

		local degree = kernelParameters.degree

		local gamma = kernelParameters.gamma

		local r = kernelParameters.r

		local scaledFeatureMatrix = AqwamTensorLibrary:multiply(featureMatrix, gamma)

		local addedFeatureMatrix = AqwamTensorLibrary:add(scaledFeatureMatrix, r)

		return AqwamTensorLibrary:power(addedFeatureMatrix, degree)

	end,

	["RadialBasisFunction"] = function(featureMatrix, kernelParameters)

		local sigma = kernelParameters.sigma

		local squaredFeatureMatrix = AqwamTensorLibrary:power(featureMatrix, 2)

		local squaredSigmaVector = AqwamTensorLibrary:power(sigma, 2)

		local multipliedSquaredSigmaVector = AqwamTensorLibrary:multiply(-2, squaredSigmaVector)

		local zMatrix = AqwamTensorLibrary:divide(squaredFeatureMatrix, multipliedSquaredSigmaVector)

		return AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

	end,

	["Sigmoid"] = function(featureMatrix, kernelParameters)

		local gamma = kernelParameters.gamma

		local r = kernelParameters.r

		local kernelMappingMatrixPart1 = AqwamTensorLibrary:multiply(gamma, featureMatrix)

		local kernelMappingMatrixPart2 = AqwamTensorLibrary:add(kernelMappingMatrixPart1, r)

		local kernelMappingMatrix = AqwamTensorLibrary:applyFunction(math.tanh, kernelMappingMatrixPart2)

		return kernelMappingMatrix

	end,

	["Cosine"] = function(featureMatrix, kernelParameters)

		local zeroMatrix = AqwamTensorLibrary:createTensor({1, #featureMatrix[1]}, 0)

		local distanceMatrix = createDistanceMatrix("Euclidean", featureMatrix, zeroMatrix)

		local kernelMappingMatrix = AqwamTensorLibrary:divide(featureMatrix, distanceMatrix)

		return kernelMappingMatrix

	end,

}

local kernelFunctionList = {

	["Linear"] = function(featureMatrix)

		local kernelMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, AqwamTensorLibrary:transpose(featureMatrix))

		return kernelMatrix

	end,

	["Polynomial"] = function(featureMatrix, kernelParameters)

		local degree = kernelParameters.degree

		local gamma = kernelParameters.gamma

		local r = kernelParameters.r

		local dotProductedFeatureMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, AqwamTensorLibrary:transpose(featureMatrix))

		local scaledDotProductedFeatureMatrix = AqwamTensorLibrary:multiply(dotProductedFeatureMatrix, gamma)

		local addedFeatureMatrix = AqwamTensorLibrary:add(scaledDotProductedFeatureMatrix, r)

		local kernelMatrix = AqwamTensorLibrary:power(addedFeatureMatrix, degree)

		return kernelMatrix

	end,

	["RadialBasisFunction"] = function(featureMatrix, kernelParameters)

		local sigma = kernelParameters.sigma

		local distanceMatrix = createDistanceMatrix("Euclidean", featureMatrix, featureMatrix)

		local squaredDistanceMatrix = AqwamTensorLibrary:power(distanceMatrix, 2)

		local sigmaSquaredVector = AqwamTensorLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamTensorLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamTensorLibrary:divide(squaredDistanceMatrix, multipliedSigmaSquaredVector)

		local kernelMatrix = AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

		return kernelMatrix

	end,

	["Sigmoid"] = function(featureMatrix, kernelParameters)

		local gamma = kernelParameters.gamma

		local r = kernelParameters.r

		local dotProductedFeatureMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, AqwamTensorLibrary:transpose(featureMatrix))

		local kernelMatrixPart1 = AqwamTensorLibrary:multiply(gamma, dotProductedFeatureMatrix)

		local kernelMatrixPart2 = AqwamTensorLibrary:add(kernelMatrixPart1, r)

		local kernelMatrix = AqwamTensorLibrary:applyFunction(math.tanh, kernelMatrixPart2)

		return kernelMatrix

	end,

	["Cosine"] = function(featureMatrix, kernelParameters)

		local zeroMatrix = AqwamTensorLibrary:createTensor({1, #featureMatrix[1]}, 0)

		local distanceMatrix = createDistanceMatrix("Euclidean", featureMatrix, zeroMatrix)

		local kernelMappingMatrix = AqwamTensorLibrary:divide(featureMatrix, distanceMatrix)

		local kernelMatrix = AqwamTensorLibrary:dotProduct(kernelMappingMatrix, AqwamTensorLibrary:transpose(kernelMappingMatrix))

		return kernelMatrix

	end,

}

local function calculatePMatrix(featureMatrix, kernelFunction, kernelParameters)
	
	local kernelFunction = kernelFunctionList[kernelFunction]
	
	if (not kernelFunction) then error("Invalid kernel function.") end
	
	local kernelMatrix = kernelFunction(featureMatrix, kernelParameters)

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local pMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, kernelMatrix, featureMatrix)

	pMatrix = AqwamTensorLibrary:inverse(pMatrix)

	pMatrix = AqwamTensorLibrary:dotProduct(pMatrix, transposedFeatureMatrix)

	return pMatrix

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local oneVector = AqwamTensorLibrary:createTensor({#labelVector, 1}, 1)
	
	local marginVector = AqwamTensorLibrary:multiply(labelVector, hypothesisVector)
	
	local hingeVector = AqwamTensorLibrary:subtract(oneVector, marginVector)
	
	local costVector = AqwamTensorLibrary:applyFunction(hingeFunction, hingeVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = (self.cValue * totalCost) / #labelVector

	return averageCost

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)
	
	local mapFunction = mappingList[self.kernelFunction]

	if (not mapFunction) then error("Invalid kernel function.") end

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	hypothesisVector = mapFunction(hypothesisVector)
	
	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix
	
	local pMatrix = self.pMatrix or calculatePMatrix(featureMatrix, self.kernelFunction, self.kernelParameters)

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(pMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end
	
	local ModelParameters = self.ModelParameters
	
	local Regularizer = self.Regularizer
	
	local learningRate = self.learningRate

	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

		lossFunctionDerivativeVector = AqwamTensorLibrary:add(lossFunctionDerivativeVector, regularizationDerivatives)

	end

	lossFunctionDerivativeVector = AqwamTensorLibrary:divide(lossFunctionDerivativeVector, numberOfData)

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, lossFunctionDerivativeVector)

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel, SupportVectorMachineIterativeReweightedLeastSquaresVariantModel)
	
	NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel:setName("SupportVectorMachineIterativeReweightedLeastSquaresVariant")

	NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel.cValue = parameterDictionary.cValue or defaultCValue

	NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel.Regularizer = parameterDictionary.Regularizer
	
	NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel.kernelFunction = parameterDictionary.kernelFunction or defaultKernelFunction

	NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel.kernelParameters = {

		degree = parameterDictionary.degree or defaultDegree,

		gamma = parameterDictionary.gamma or defaultGamma,

		sigma = parameterDictionary.sigma or defaultSigma,

		r = parameterDictionary.r or defaultR

	}

	return NewSupportVectorMachineIterativeReweightedLeastSquaresVariantModel

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue

	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
	self.pMatrix = calculatePMatrix(featureMatrix, self.kernelFunction, self.kernelParameters)

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local hypothesisVector = self:calculateHypothesisVector(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisVector, labelVector)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local marginVector = AqwamTensorLibrary:multiply(labelVector, hypothesisVector)
		
		local misclassifiedMaskVector = AqwamTensorLibrary:applyFunction(misclassificationMaskFunction, marginVector)
		
		local lossGradientVector = AqwamTensorLibrary:multiply(-cValue, labelVector, misclassifiedMaskVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	self.pMatrix = nil

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	return costArray

end

function SupportVectorMachineIterativeReweightedLeastSquaresVariantModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	if (returnOriginalOutput) then return predictedVector end

	return AqwamTensorLibrary:applyFunction(seperatorFunction, predictedVector)

end

return SupportVectorMachineIterativeReweightedLeastSquaresVariantModel
