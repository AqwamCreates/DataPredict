--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.01

local defaultCvalue = 1

local defaultKernelFunction = "Linear"

local defaultGamma = 1

local defaultDegree = 3

local defaultSigma = 1

local defaultR = 0

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local part3 = AqwamMatrixLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,
	
	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamMatrixLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamMatrixLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamMatrixLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamMatrixLibrary:sum(x2MagnitudePart1)

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

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData1, numberOfData2)
	
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
		
		local scaledFeatureMatrix = AqwamMatrixLibrary:multiply(featureMatrix, gamma)
		
		local addedFeatureMatrix = AqwamMatrixLibrary:add(scaledFeatureMatrix, r)

		return AqwamMatrixLibrary:power(addedFeatureMatrix, degree)

	end,

	["RadialBasisFunction"] = function(featureMatrix, kernelParameters)
		
		local sigma = kernelParameters.sigma or defaultSigma

		local squaredFeatureMatrix = AqwamMatrixLibrary:power(featureMatrix, 2)

		local squaredSigmaVector = AqwamMatrixLibrary:power(sigma, 2)

		local multipliedSquaredSigmaVector = AqwamMatrixLibrary:multiply(-2, squaredSigmaVector)

		local zMatrix = AqwamMatrixLibrary:divide(squaredFeatureMatrix, multipliedSquaredSigmaVector)

		return AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

	end,
	
	["Sigmoid"] = function(featureMatrix, kernelParameters)

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR
		
		local kernelMappingMatrixPart1 = AqwamMatrixLibrary:multiply(gamma, featureMatrix)

		local kernelMappingMatrixPart2 = AqwamMatrixLibrary:add(kernelMappingMatrixPart1, r)

		local kernelMappingMatrix = AqwamMatrixLibrary:applyFunction(math.tanh, kernelMappingMatrixPart2)
		
		return kernelMappingMatrix

	end,
	
	["Cosine"] = function(featureMatrix, kernelParameters)
		
		local zeroMatrix = AqwamMatrixLibrary:createMatrix(1, #featureMatrix[1])

		local distanceMatrix = createDistanceMatrix(featureMatrix, zeroMatrix, "Euclidean")

		local kernelMappingMatrix = AqwamMatrixLibrary:divide(featureMatrix, distanceMatrix)

		return kernelMappingMatrix

	end,

}

local kernelFunctionList = {

	["Linear"] = function(featureMatrix)

		local kernelMatrix = AqwamMatrixLibrary:dotProduct(featureMatrix, AqwamMatrixLibrary:transpose(featureMatrix))

		return kernelMatrix

	end,

	["Polynomial"] = function(featureMatrix, kernelParameters)

		local degree = kernelParameters.degree or defaultDegree

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR

		local dotProductedFeatureMatrix = AqwamMatrixLibrary:dotProduct(featureMatrix, AqwamMatrixLibrary:transpose(featureMatrix))

		local scaledDotProductedFeatureMatrix = AqwamMatrixLibrary:multiply(dotProductedFeatureMatrix, gamma)

		local addedFeatureMatrix = AqwamMatrixLibrary:add(scaledDotProductedFeatureMatrix, r)

		local kernelMatrix = AqwamMatrixLibrary:power(addedFeatureMatrix, degree)

		return kernelMatrix

	end,

	["RadialBasisFunction"] = function(featureMatrix, kernelParameters)

		local sigma	= kernelParameters.sigma or defaultSigma

		local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, "Euclidean")

		local squaredDistanceMatrix = AqwamMatrixLibrary:power(distanceMatrix, 2)

		local sigmaSquaredVector = AqwamMatrixLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamMatrixLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamMatrixLibrary:divide(squaredDistanceMatrix, multipliedSigmaSquaredVector)

		local kernelMatrix = AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

		return kernelMatrix

	end,

	["Sigmoid"] = function(featureMatrix, kernelParameters)

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR

		local dotProductedFeatureMatrix = AqwamMatrixLibrary:dotProduct(featureMatrix, AqwamMatrixLibrary:transpose(featureMatrix))

		local kernelMatrixPart1 = AqwamMatrixLibrary:multiply(gamma, dotProductedFeatureMatrix)

		local kernelMatrixPart2 = AqwamMatrixLibrary:add(kernelMatrixPart1, r)

		local kernelMatrix = AqwamMatrixLibrary:applyFunction(math.tanh, kernelMatrixPart2)

		return kernelMatrix

	end,

	["Cosine"] = function(featureMatrix, kernelParameters)
		
		local zeroMatrix = AqwamMatrixLibrary:createMatrix(1, #featureMatrix[1])

		local distanceMatrix = createDistanceMatrix(featureMatrix, zeroMatrix, "Euclidean")

		local kernelMappingMatrix = AqwamMatrixLibrary:divide(featureMatrix, distanceMatrix)
		
		local kernelMatrix = AqwamMatrixLibrary:dotProduct(kernelMappingMatrix, AqwamMatrixLibrary:transpose(kernelMappingMatrix))

		return kernelMatrix

	end,

}

local function calculateCost(modelParameters, individualKernelMatrix, kernelMatrix, labelVector, cValue)
	
	-- The dotProduct() only takes two arguments here to reduce computational time
	
	local predictedVector = AqwamMatrixLibrary:dotProduct(individualKernelMatrix, modelParameters)
	
	local costVector = AqwamMatrixLibrary:subtract(predictedVector, labelVector)
	
	costVector = AqwamMatrixLibrary:multiply(-cValue, costVector)
	
	local transposedCostVector = AqwamMatrixLibrary:transpose(costVector)
	
	local transposedLabelVector = AqwamMatrixLibrary:transpose(labelVector)
	
	local costPart1 = AqwamMatrixLibrary:dotProduct(transposedCostVector, kernelMatrix)
	
	costPart1 = AqwamMatrixLibrary:dotProduct(costPart1, kernelMatrix)
	
	costPart1 = AqwamMatrixLibrary:dotProduct(costPart1, costVector)
	
	costPart1 /= 2
	
	local costPart2 = AqwamMatrixLibrary:dotProduct(transposedCostVector, kernelMatrix)
	
	costPart2 = AqwamMatrixLibrary:dotProduct(costPart2, labelVector)
	
	local costPart3 = AqwamMatrixLibrary:dotProduct(transposedLabelVector, labelVector)
	
	costPart3 /= 2
	
	local costPart4 = AqwamMatrixLibrary:dotProduct(transposedCostVector, kernelMatrix)
	
	costPart4 = AqwamMatrixLibrary:dotProduct(costPart4, costVector)
	
	costPart4 /= (2 * cValue)
	
	local cost = costPart1 - costPart2 + costPart3 + costPart4
	
	return cost

end

local function calculateModelParameters(modelParameters, individualkernelMatrix, labelVector, cValue)

	local predictionVector = AqwamMatrixLibrary:dotProduct(individualkernelMatrix, modelParameters) -- m x 1
	
	local errorVector = AqwamMatrixLibrary:subtract(predictionVector, labelVector) -- m x 1
	
	local transposedKernelMatrix = AqwamMatrixLibrary:transpose(individualkernelMatrix)
	
	local dotProductErrorVector = AqwamMatrixLibrary:dotProduct(transposedKernelMatrix, errorVector) -- n x m, m x 1
	
	local NewModelParameters = AqwamMatrixLibrary:multiply(-cValue, dotProductErrorVector)

	return NewModelParameters

end

function SupportVectorMachineModel.new(maxNumberOfIterations, cValue, kernelFunction, kernelParameters)

	local NewSupportVectorMachine = BaseModel.new()

	setmetatable(NewSupportVectorMachine, SupportVectorMachineModel)

	NewSupportVectorMachine.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewSupportVectorMachine.cValue = cValue or defaultCvalue

	NewSupportVectorMachine.kernelFunction = kernelFunction or defaultKernelFunction

	NewSupportVectorMachine.kernelParameters = kernelParameters or {}

	return NewSupportVectorMachine
end

function SupportVectorMachineModel:setParameters(maxNumberOfIterations, cValue, kernelFunction, kernelParameters)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.cValue = cValue or self.cValue

	self.kernelFunction = kernelFunction or self.kernelFunction

	self.kernelParameters = kernelParameters or self.kernelParameters

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

		self.ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], 1)

	end

	local cost

	local costArray = {}

	local numberOfIterations = 0

	local costFunctionDerivatives

	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)
	
	local kernelMatrix = kernelFunctionList[self.kernelFunction](featureMatrix, self.kernelParameters)
	
	repeat
		
		numberOfIterations += 1

		self:iterationWait()
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(self.ModelParameters, mappedFeatureMatrix, kernelMatrix, labelVector, self.cValue)
			
		end)

		if cost then
			
			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
		end

		self.ModelParameters = calculateModelParameters(self.ModelParameters, mappedFeatureMatrix, labelVector, self.cValue)

	until (numberOfIterations == self.maxNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then

		warn("The model diverged! Please repeat the experiment or change the argument values.")

	end

	return costArray

end

function SupportVectorMachineModel:predict(featureMatrix, returnOriginalOutput)

	local mappedFeatureMatrix = mappingList[self.kernelFunction](featureMatrix, self.kernelParameters)

	local originalPredictedVector = AqwamMatrixLibrary:dotProduct(mappedFeatureMatrix, self.ModelParameters)

	if (typeof(originalPredictedVector) == "number") then originalPredictedVector = {{originalPredictedVector}} end

	if (returnOriginalOutput == true) then return originalPredictedVector end

	local seperatorFunction = function (x) 
		
		return ((x > 0) and 1) or ((x < 0) and -1) or 0
		
	end

	local predictedVector = AqwamMatrixLibrary:applyFunction(seperatorFunction, originalPredictedVector)

	return predictedVector

end

return SupportVectorMachineModel
