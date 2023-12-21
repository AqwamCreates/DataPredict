--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, BaseModel)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.01

local defaultCvalue = 1

local defaultTargetCost = 0

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

}

local mappingList = {

	["Linear"] = function(X)

		return X

	end,

	["Polynomial"] = function(X, kernelParameters)
		
		local degree = kernelParameters.degree or defaultDegree
		
		local gamma = kernelParameters.gamma or defaultGamma
		
		local r = kernelParameters.r or defaultR
		
		local scaledX = AqwamMatrixLibrary:multiply(X, gamma)
		
		local addedX = AqwamMatrixLibrary:add(scaledX, r)

		return AqwamMatrixLibrary:power(addedX, degree)

	end,

	["RadialBasisFunction"] = function(X, kernelParameters)
		
		local sigma = kernelParameters.sigma or defaultSigma

		local XSquaredVector = AqwamMatrixLibrary:power(X, 2)

		local sigmaSquaredVector = AqwamMatrixLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamMatrixLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamMatrixLibrary:divide(XSquaredVector, multipliedSigmaSquaredVector)

		return AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

	end,

	["CosineSimilarity"] = function(X)

		local XSquaredVector = AqwamMatrixLibrary:power(X, 2)

		local normXVector = AqwamMatrixLibrary:applyFunction(math.sqrt, XSquaredVector)

		return AqwamMatrixLibrary:divide(X, normXVector)

	end,
	
	["Sigmoid"] = function(X, kernelParameters)

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR
		
		local kernelMatrixPart1 = AqwamMatrixLibrary:multiply(gamma, X)

		local kernelMatrixPart2 = AqwamMatrixLibrary:add(kernelMatrixPart1, r)

		local kernelMatrix = AqwamMatrixLibrary:applyFunction(math.tanh, kernelMatrixPart2)
		
		return kernelMatrix

	end,

}

local function calculateDistance(vector1, vector2, distanceFunction)

	return distanceFunctionList[distanceFunction](vector1, vector2) 

end

local function calculateMapping(x, kernelFunction, kernelParameters)

	return mappingList[kernelFunction](x, kernelParameters)

end

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData1, numberOfData2)

	for i = 1, numberOfData1, 1 do

		for j = 1, numberOfData2, 1 do

			distanceMatrix[i][j] = calculateDistance({matrix1[i]}, {matrix2[j]}, distanceFunction)

		end

	end

	return distanceMatrix

end

local kernelFunctionList = {

	["Linear"] = function(X)

		local kernelMatrix = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))

		return kernelMatrix

	end,

	["Polynomial"] = function(X, kernelParameters)
		
		local degree = kernelParameters.degree or defaultDegree

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR

		local dotProductedX = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))

		local scaledDotProductedX = AqwamMatrixLibrary:multiply(dotProductedX, gamma)
		
		local addedX = AqwamMatrixLibrary:add(scaledDotProductedX, r)

		local kernelMatrix = AqwamMatrixLibrary:power(addedX, degree)

		return kernelMatrix

	end,

	["RadialBasisFunction"] = function(X, kernelParameters)
		
		local sigma	= kernelParameters.sigma or defaultSigma

		local distanceMatrix = createDistanceMatrix(X, X, "Euclidean")
		
		local squaredDistanceMatrix = AqwamMatrixLibrary:power(distanceMatrix, 2)

		local sigmaSquaredVector = AqwamMatrixLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamMatrixLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamMatrixLibrary:divide(squaredDistanceMatrix, multipliedSigmaSquaredVector)

		local kernelMatrix = AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

		return kernelMatrix

	end,

	["CosineSimilarity"] = function(X)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))

		local distanceMatrix = calculateDistance(X, X, "Euclidean")

		local normX = AqwamMatrixLibrary:power(distanceMatrix, 2)

		local kernelMatrix = AqwamMatrixLibrary:divide(dotProductedX, normX)

		return kernelMatrix

	end,
	
	["Sigmoid"] = function(X, kernelParameters)

		local gamma = kernelParameters.gamma or defaultGamma

		local r = kernelParameters.r or defaultR
		
		local dotProductedX = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))
		
		local kernelMatrixPart1 = AqwamMatrixLibrary:multiply(gamma, dotProductedX)
		
		local kernelMatrixPart2 = AqwamMatrixLibrary:add(kernelMatrixPart1, r)
		
		local kernelMatrix = AqwamMatrixLibrary:applyFunction(math.tanh, kernelMatrixPart2)
		
		return kernelMatrix
		
	end,

}

local function calculateKernel(x, kernelFunction, kernelParameters)

	return kernelFunctionList[kernelFunction](x, kernelParameters)

end

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

function SupportVectorMachineModel.new(maxNumberOfIterations, cValue, targetCost, kernelFunction, kernelParameters)

	local NewSupportVectorMachine = BaseModel.new()

	setmetatable(NewSupportVectorMachine, SupportVectorMachineModel)

	NewSupportVectorMachine.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewSupportVectorMachine.cValue = cValue or defaultCvalue

	NewSupportVectorMachine.targetCost = targetCost or defaultTargetCost

	NewSupportVectorMachine.kernelFunction = kernelFunction or defaultKernelFunction

	NewSupportVectorMachine.kernelParameters = kernelParameters or {}

	NewSupportVectorMachine.Optimizer = nil

	return NewSupportVectorMachine
end

function SupportVectorMachineModel:setParameters(maxNumberOfIterations, cValue, targetCost, kernelFunction, kernelParameters)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.cValue = cValue or self.cValue

	self.targetCost = targetCost or self.targetCost

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

	local mappedFeatureMatrix  = calculateMapping(featureMatrix, self.kernelFunction, self.kernelParameters)
	
	local kernelMatrix  = calculateKernel(featureMatrix, self.kernelFunction, self.kernelParameters)
	
	repeat
		
		numberOfIterations += 1

		self:iterationWait()
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(self.ModelParameters, mappedFeatureMatrix, kernelMatrix, labelVector, self.cValue)
			
		end)

		if cost then
			
			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
			if (math.abs(cost) <= self.targetCost) then break end
			
		end

		self.ModelParameters = calculateModelParameters(self.ModelParameters, mappedFeatureMatrix, labelVector, self.cValue)

	until (numberOfIterations == self.maxNumberOfIterations)

	if (cost == math.huge) then

		warn("The model diverged! Please repeat the experiment or change the argument values.")

	end

	return costArray

end

function SupportVectorMachineModel:predict(featureMatrix, returnOriginalOutput)

	local mappedFeatureMatrix = calculateMapping(featureMatrix, self.kernelFunction, self.kernelParameters)

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
