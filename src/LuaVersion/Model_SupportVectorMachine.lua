--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, IterativeMethodBaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.01

local defaultCvalue = 1

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
		
		local degree = kernelParameters.degree or defaultDegree
		
		local gamma = kernelParameters.gamma or defaultGamma
		
		local r = kernelParameters.r or defaultR
		
		local scaledFeatureMatrix = AqwamTensorLibrary:multiply(featureMatrix, gamma)
		
		local addedFeatureMatrix = AqwamTensorLibrary:add(scaledFeatureMatrix, r)

		return AqwamTensorLibrary:power(addedFeatureMatrix, degree)

	end,

	["RadialBasisFunction"] = function(featureMatrix, kernelParameters)
		
		local sigma = kernelParameters.sigma or defaultSigma

		local squaredFeatureMatrix = AqwamTensorLibrary:power(featureMatrix, 2)

		local squaredSigmaVector = AqwamTensorLibrary:power(sigma, 2)

		local multipliedSquaredSigmaVector = AqwamTensorLibrary:multiply(-2, squaredSigmaVector)

		local zMatrix = AqwamTensorLibrary:divide(squaredFeatureMatrix, multipliedSquaredSigmaVector)

		return AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

	end,
	
	["Sigmoid"] = function(featureMatrix, kernelParameters)

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR
		
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

		local degree = kernelParameters.degree or defaultDegree

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR

		local dotProductedFeatureMatrix = AqwamTensorLibrary:dotProduct(featureMatrix, AqwamTensorLibrary:transpose(featureMatrix))

		local scaledDotProductedFeatureMatrix = AqwamTensorLibrary:multiply(dotProductedFeatureMatrix, gamma)

		local addedFeatureMatrix = AqwamTensorLibrary:add(scaledDotProductedFeatureMatrix, r)

		local kernelMatrix = AqwamTensorLibrary:power(addedFeatureMatrix, degree)

		return kernelMatrix

	end,

	["RadialBasisFunction"] = function(featureMatrix, kernelParameters)

		local sigma = kernelParameters.sigma or defaultSigma

		local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, "Euclidean")

		local squaredDistanceMatrix = AqwamTensorLibrary:power(distanceMatrix, 2)

		local sigmaSquaredVector = AqwamTensorLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamTensorLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamTensorLibrary:divide(squaredDistanceMatrix, multipliedSigmaSquaredVector)

		local kernelMatrix = AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

		return kernelMatrix

	end,

	["Sigmoid"] = function(featureMatrix, kernelParameters)

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR

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

local function calculateCost(modelParameters, individualKernelMatrix, kernelMatrix, labelVector, cValue)
	
	-- The dotProduct() only takes two arguments here to reduce computational time
	
	local predictedVector = AqwamTensorLibrary:dotProduct(individualKernelMatrix, modelParameters)
	
	local costVector = AqwamTensorLibrary:subtract(predictedVector, labelVector)
	
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

local function calculateModelParameters(modelParameters, individualkernelMatrix, labelVector, cValue)

	local predictionVector = AqwamTensorLibrary:dotProduct(individualkernelMatrix, modelParameters) -- m x 1
	
	local errorVector = AqwamTensorLibrary:subtract(predictionVector, labelVector) -- m x 1
	
	local transposedKernelMatrix = AqwamTensorLibrary:transpose(individualkernelMatrix)
	
	local dotProductErrorVector = AqwamTensorLibrary:dotProduct(transposedKernelMatrix, errorVector) -- n x m, m x 1
	
	local NewModelParameters = AqwamTensorLibrary:multiply(-cValue, dotProductErrorVector)

	return NewModelParameters

end

function SupportVectorMachineModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorMachine = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorMachine, SupportVectorMachineModel)
	
	NewSupportVectorMachine:setName("SupportVectorMachine")
	
	NewSupportVectorMachine.cValue = parameterDictionary.cValue or defaultCvalue

	NewSupportVectorMachine.kernelFunction = parameterDictionary.kernelFunction or defaultKernelFunction

	NewSupportVectorMachine.kernelParameters = parameterDictionary.kernelParameters or {}

	return NewSupportVectorMachine
end

function SupportVectorMachineModel:setCValue(cValue)

	self.cValue = cValue or self.cValue

end

function SupportVectorMachineModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then

		error("The feature matrix and the label vector do not contain the same number of rows!")

	end

	if (self.ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then

			error("The number of features is not the same as the model parameters!")

		end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end

	local cost

	local costArray = {}

	local numberOfIterations = 0
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations

	local costFunctionDerivatives

	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)
	
	local kernelMatrix = kernelFunctionList[self.kernelFunction](featureMatrix, self.kernelParameters)
	
	repeat
		
		numberOfIterations = numberOfIterations + 1

		self:iterationWait()
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(self.ModelParameters, mappedFeatureMatrix, kernelMatrix, labelVector, self.cValue)
			
		end)

		if cost then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)
			
		end

		self.ModelParameters = calculateModelParameters(self.ModelParameters, mappedFeatureMatrix, labelVector, self.cValue)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then

		warn("The model diverged! Please repeat the experiment or change the argument values.")

	end

	return costArray

end

function SupportVectorMachineModel:predict(featureMatrix, returnOriginalOutput)

	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)

	local originalPredictedVector = AqwamTensorLibrary:dotProduct(mappedFeatureMatrix, self.ModelParameters)

	if (typeof(originalPredictedVector) == "number") then originalPredictedVector = {{originalPredictedVector}} end

	if (returnOriginalOutput) then return originalPredictedVector end

	local predictedVector = AqwamTensorLibrary:applyFunction(seperatorFunction, originalPredictedVector)

	return predictedVector

end

return SupportVectorMachineModel