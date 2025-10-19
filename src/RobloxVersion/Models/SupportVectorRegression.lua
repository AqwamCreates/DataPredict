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

local IterativeMethodBaseModel = require(script.Parent.IterativeMethodBaseModel)

SupportVectorRegressionModel = {}

SupportVectorRegressionModel.__index = SupportVectorRegressionModel

setmetatable(SupportVectorRegressionModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultCvalue = 1

local defaultEpsilon = 1

local defaultKernelFunction = "Linear"

local defaultGamma = 1

local defaultDegree = 3

local defaultSigma = 1

local defaultR = 0

local seperatorFunction = function (x) 

	return ((x > 0) and 1) or ((x < 0) and -1) or 0

end

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		part1 = AqwamTensorLibrary:applyFunction(math.abs, part1)

		local distance = AqwamTensorLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		local part2 = AqwamTensorLibrary:power(part1, 2)

		local part3 = AqwamTensorLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,

	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamTensorLibrary:dotProduct(x1, AqwamTensorLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamTensorLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamTensorLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2, 2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData1, numberOfData2})

	local distanceFunctionToApply = distanceFunctionList[distanceFunction]

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

		local distanceMatrix = createDistanceMatrix(featureMatrix, zeroMatrix, "Euclidean")

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

		local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, "Euclidean")

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

		local distanceMatrix = createDistanceMatrix(featureMatrix, zeroMatrix, "Euclidean")

		local kernelMappingMatrix = AqwamTensorLibrary:divide(featureMatrix, distanceMatrix)

		local kernelMatrix = AqwamTensorLibrary:dotProduct(kernelMappingMatrix, AqwamTensorLibrary:transpose(kernelMappingMatrix))

		return kernelMatrix

	end,

}

local function calculateCost(modelParameters, individualKernelMatrix, kernelMatrix, labelVector, cValue, epsilon)

	-- The dotProduct() only takes two arguments here to reduce computational time

	local predictedVector = AqwamTensorLibrary:dotProduct(individualKernelMatrix, modelParameters)

	local errorVector = AqwamTensorLibrary:subtract(predictedVector, labelVector)
	
	local positiveSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, errorValue - epsilon) end, errorVector)

	local negativeSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, -errorValue - epsilon) end, errorVector)

	local costVector = AqwamTensorLibrary:add(positiveSlackVariableVector, negativeSlackVariableVector)

	costVector = AqwamTensorLibrary:multiply(-cValue, costVector)

	local transposedCostVector = AqwamTensorLibrary:transpose(costVector)

	local transposedLabelVector = AqwamTensorLibrary:transpose(labelVector)

	local costPart1 = AqwamTensorLibrary:dotProduct(transposedCostVector, kernelMatrix)

	costPart1 = AqwamTensorLibrary:dotProduct(costPart1, kernelMatrix)

	costPart1 = AqwamTensorLibrary:dotProduct(costPart1, costVector)

	costPart1 = costPart1 / 2

	local costPart2 = AqwamTensorLibrary:dotProduct(transposedCostVector, kernelMatrix)

	costPart2 = AqwamTensorLibrary:dotProduct(costPart2, labelVector)

	local costPart3 = AqwamTensorLibrary:dotProduct(transposedLabelVector, labelVector)

	costPart3 = costPart3 / 2

	local costPart4 = AqwamTensorLibrary:dotProduct(transposedCostVector, kernelMatrix)

	costPart4 = AqwamTensorLibrary:dotProduct(costPart4, costVector)

	costPart4 = costPart4 / (2 * cValue)

	local cost = costPart1 - costPart2 + costPart3 + costPart4

	return cost

end

local function calculateModelParameters(modelParameters, individualKernelMatrix, labelVector, cValue, epsilon)

	local predictionVector = AqwamTensorLibrary:dotProduct(individualKernelMatrix, modelParameters) -- m x 1

	local errorVector = AqwamTensorLibrary:subtract(predictionVector, labelVector) -- m x 1
	
	local positiveSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, errorValue - epsilon) end, errorVector)

	local negativeSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, -errorValue - epsilon) end, errorVector)
	
	local slackVariableVector = AqwamTensorLibrary:add(positiveSlackVariableVector, negativeSlackVariableVector)
	
	local transposedIndividualKernelMatrix = AqwamTensorLibrary:transpose(individualKernelMatrix)

	local dotProductErrorVector = AqwamTensorLibrary:dotProduct(transposedIndividualKernelMatrix, slackVariableVector) -- n x m, m x 1

	local NewModelParameters = AqwamTensorLibrary:multiply(-cValue, dotProductErrorVector)

	return NewModelParameters

end

function SupportVectorRegressionModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorRegression = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorRegression, SupportVectorRegressionModel)

	NewSupportVectorRegression:setName("SupportVectorRegression")

	NewSupportVectorRegression.cValue = parameterDictionary.cValue or defaultCvalue
	
	NewSupportVectorRegression.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewSupportVectorRegression.kernelFunction = parameterDictionary.kernelFunction or defaultKernelFunction

	NewSupportVectorRegression.kernelParameters = {

		degree = parameterDictionary.degree or defaultDegree,

		gamma = parameterDictionary.gamma or defaultGamma,

		sigma = parameterDictionary.sigma or defaultSigma,

		r = parameterDictionary.r or defaultR

	}

	return NewSupportVectorRegression
end

function SupportVectorRegressionModel:setCValue(cValue)

	self.cValue = cValue or self.cValue

end

function SupportVectorRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then

		error("The feature matrix and the label vector do not contain the same number of rows!")

	end

	local numberOfFeatures = #featureMatrix[1]

	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (numberOfFeatures ~= #ModelParameters) then

			error("The number of features is not the same as the model parameters!")

		end

	else

		ModelParameters = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end

	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue
	
	local epsilon = self.epsilon
	
	local kernelFunction = self.kernelFunction
	
	local kernelParameters = self.kernelParameters
	
	local ModelParameters = self.ModelParameters

	local mappedFeatureMatrix = mappingList[kernelFunction](featureMatrix, kernelParameters)

	local kernelMatrix = kernelFunctionList[kernelFunction](featureMatrix, kernelParameters)

	local costArray = {}

	local numberOfIterations = 0
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return calculateCost(ModelParameters, mappedFeatureMatrix, kernelMatrix, labelVector, cValue, epsilon)

		end)

		if cost then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		ModelParameters = calculateModelParameters(ModelParameters, mappedFeatureMatrix, labelVector, cValue, epsilon)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	self.ModelParameters = ModelParameters

	return costArray

end

function SupportVectorRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then

		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = ModelParameters

	end

	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)

	local predictedVector = AqwamTensorLibrary:dotProduct(mappedFeatureMatrix, ModelParameters)

	return predictedVector

end

return SupportVectorRegressionModel
