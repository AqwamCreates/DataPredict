--[[

	--------------------------------------------------------------------

	Aqwam's Regression, Deep And Reinforcement Learning Library (DataPredict)

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

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

local Solvers = script.Parent.Parent.Solvers

local SupportVectorRegressionGradientVariantModel = {}

SupportVectorRegressionGradientVariantModel.__index = SupportVectorRegressionGradientVariantModel

setmetatable(SupportVectorRegressionGradientVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCValue = 1

local defaultEpsilon = 1

local defaultKernelFunction = "Linear"

local defaultGamma = 1

local defaultDegree = 3

local defaultSigma = 1

local defaultR = 0

local defaultSolver = "GaussNewton"

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

function SupportVectorRegressionGradientVariantModel:calculateCost(hypothesisVector, labelVector, hasBias)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local epsilon = self.epsilon
	
	local errorVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)
	
	local positiveSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, errorValue - epsilon) end, errorVector)

	local negativeSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, -errorValue - epsilon) end, errorVector)

	local slackVariableVector = AqwamTensorLibrary:add(positiveSlackVariableVector, negativeSlackVariableVector)

	local totalCost = AqwamTensorLibrary:sum(slackVariableVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters, hasBias) end

	local averageCost = (self.cValue * totalCost) / #labelVector

	return averageCost

end

function SupportVectorRegressionGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorRegressionGradientVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local lossFunctionDerivativeVector = self.Solver:calculate(self.ModelParameters, self.featureMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function SupportVectorRegressionGradientVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end
	
	local ModelParameters = self.ModelParameters
	
	local Regularizer = self.Regularizer
	
	local Optimizer = self.Optimizer
	
	local learningRate = self.learningRate

	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters, hasBias)

		lossFunctionDerivativeVector = AqwamTensorLibrary:add(lossFunctionDerivativeVector, regularizationDerivatives)

	end

	lossFunctionDerivativeVector = AqwamTensorLibrary:divide(lossFunctionDerivativeVector, numberOfData)

	if (Optimizer) then 

		lossFunctionDerivativeVector = Optimizer:calculate(learningRate, lossFunctionDerivativeVector, ModelParameters) 

	else

		lossFunctionDerivativeVector = AqwamTensorLibrary:multiply(learningRate, lossFunctionDerivativeVector)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, lossFunctionDerivativeVector)

end

function SupportVectorRegressionGradientVariantModel:update(lossGradientVector, hasBias, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (clearAllMatrices) then 

		self.featureMatrix = nil

		self.lossFunctionDerivativeVector = nil

	end

end

function SupportVectorRegressionGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorRegressionGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorRegressionGradientVariantModel, SupportVectorRegressionGradientVariantModel)
	
	NewSupportVectorRegressionGradientVariantModel:setName("SupportVectorRegressionGradientVariant")

	NewSupportVectorRegressionGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorRegressionGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue
	
	NewSupportVectorRegressionGradientVariantModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewSupportVectorRegressionGradientVariantModel.Optimizer = parameterDictionary.Optimizer

	NewSupportVectorRegressionGradientVariantModel.Regularizer = parameterDictionary.Regularizer
	
	NewSupportVectorRegressionGradientVariantModel.Solver = parameterDictionary.Solver or require(Solvers[defaultSolver]).new({isLinear = true})
	
	NewSupportVectorRegressionGradientVariantModel.kernelFunction = parameterDictionary.kernelFunction or defaultKernelFunction

	NewSupportVectorRegressionGradientVariantModel.kernelParameters = {

		degree = parameterDictionary.degree or defaultDegree,

		gamma = parameterDictionary.gamma or defaultGamma,

		sigma = parameterDictionary.sigma or defaultSigma,

		r = parameterDictionary.r or defaultR

	}

	return NewSupportVectorRegressionGradientVariantModel

end

function SupportVectorRegressionGradientVariantModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function SupportVectorRegressionGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorRegressionGradientVariantModel:setSolver(Solver)

	self.Solver = Solver

end

function SupportVectorRegressionGradientVariantModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue
	
	local epsilon = self.epsilon

	local Optimizer = self.Optimizer
	
	local functionToApply = function(errorValue) return ((errorValue > epsilon) and (errorValue - epsilon)) or ((errorValue < -epsilon) and (errorValue + epsilon)) or 0 end
	
	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)
	
	local hasBias = self:checkIfFeatureMatrixHasBias(featureMatrix)
	
	local costArray = {}

	local numberOfIterations = 0
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local hypothesisVector = self:calculateHypothesisVector(mappedFeatureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisVector, labelVector, hasBias)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
		local errorVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		local lossGradientVector = AqwamTensorLibrary:applyFunction(functionToApply, errorVector)
		
		lossGradientVector = AqwamTensorLibrary:multiply(cValue, lossGradientVector)

		self:update(lossGradientVector, hasBias, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end
	
	if (self.autoResetSolvers) then self.Solver:reset() end

	return costArray

end

function SupportVectorRegressionGradientVariantModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end
	
	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)

	local predictedVector = AqwamTensorLibrary:dotProduct(mappedFeatureMatrix, ModelParameters)

	return predictedVector

end

return SupportVectorRegressionGradientVariantModel
