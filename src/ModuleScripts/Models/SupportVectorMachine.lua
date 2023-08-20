local BaseModel = require(script.Parent.BaseModel)

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.01

local defaultCvalue = 1

local defaultTargetCost = 0

local defaultKernelFunction = "linear"

local defaultGamma = 1

local defaultDegree = 3

local defaultSigma = 1

local defaultR = 0

local distanceFunctionList = {

	["manhattan"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["euclidean"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local part3 = AqwamMatrixLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,

}

local mappingList = {

	["linear"] = function(X)

		return X

	end,

	["polynomial"] = function(X, degree, gamma, r)
		
		local scaledX = AqwamMatrixLibrary:multiply(X, gamma)
		
		local addedX = AqwamMatrixLibrary:add(scaledX, r)

		return AqwamMatrixLibrary:power(addedX, degree)

	end,

	["radialBasisFunction"] = function(X, sigma)

		local XSquaredVector = AqwamMatrixLibrary:power(X, 2)

		local sigmaSquaredVector = AqwamMatrixLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamMatrixLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamMatrixLibrary:divide(XSquaredVector, multipliedSigmaSquaredVector)

		return AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

	end,

	["cosineSimilarity"] = function(X)

		local XSquaredVector = AqwamMatrixLibrary:power(X, 2)

		local normXVector = AqwamMatrixLibrary:applyFunction(math.sqrt, XSquaredVector)

		return AqwamMatrixLibrary:divide(X, normXVector)

	end,

}

local function calculateDistance(vector1, vector2, distanceFunction)

	return distanceFunctionList[distanceFunction](vector1, vector2) 

end

local function calculateMapping(x, kernelFunction, kernelParameters)

	if (kernelFunction == "linear") or (kernelFunction == "cosineSimilarity") then

		return mappingList[kernelFunction](x)

	elseif (kernelFunction == "polynomial") then

		local degree = kernelParameters.degree or defaultDegree
		
		local gamma = kernelParameters.gamma or defaultGamma
		
		local r = kernelParameters.r or defaultR

		return mappingList[kernelFunction](x, degree, gamma, r)

	elseif (kernelFunction == "radialBasisFunction") then

		local sigma = kernelParameters.sigma or defaultSigma

		return mappingList[kernelFunction](x, sigma)

	end

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

	["linear"] = function(X)

		local kernelMatrix = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))

		return kernelMatrix

	end,

	["polynomial"] = function(X, degree, gamma, r)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))

		local scaledDotProductedX = AqwamMatrixLibrary:multiply(dotProductedX, gamma)
		
		local addedX = AqwamMatrixLibrary:add(scaledDotProductedX, r)

		local kernelMatrix = AqwamMatrixLibrary:power(addedX, degree)

		return kernelMatrix

	end,

	["radialBasisFunction"] = function(X, sigma)

		local distanceMatrix = createDistanceMatrix(X, X, "euclidean")

		local sigmaSquaredVector = AqwamMatrixLibrary:power(sigma, 2)

		local multipliedSigmaSquaredVector = AqwamMatrixLibrary:multiply(-2, sigmaSquaredVector)

		local zMatrix = AqwamMatrixLibrary:divide(distanceMatrix, multipliedSigmaSquaredVector)

		local kernelMatrix = AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

		return kernelMatrix

	end,

	["cosineSimilarity"] = function(X)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(X, AqwamMatrixLibrary:transpose(X))

		local distanceMatrix = calculateDistance(X, X, "euclidean")

		local normX = AqwamMatrixLibrary:power(distanceMatrix, 2)

		local kernelMatrix = AqwamMatrixLibrary:divide(dotProductedX, normX)

		return kernelMatrix

	end,

}

local function calculateKernel(x, kernelFunction, kernelParameters)

	if (kernelFunction == "linear") or (kernelFunction == "cosineSimilarity") then

		return kernelFunctionList[kernelFunction](x)

	elseif (kernelFunction == "polynomial") then

		local degree = kernelParameters.degree or defaultDegree

		local gamma = kernelParameters.gamma or defaultGamma
		
		local r = kernelParameters.r or defaultR

		return kernelFunctionList[kernelFunction](x, degree, gamma, r)

	elseif (kernelFunction == "radialBasisFunction") then

		local sigma = kernelParameters.sigma or defaultSigma

		return kernelFunctionList[kernelFunction](x, sigma)

	end

end

local function calculateCost(modelParameters, individualKernelMatrix, kernelMatrix, labelVector, cValue)

	local numberOfData = #labelVector

	local prediction = AqwamMatrixLibrary:dotProduct(individualKernelMatrix, modelParameters)

	local costVector = AqwamMatrixLibrary:subtract(prediction, labelVector)
	
	local transposedCostVector = AqwamMatrixLibrary:transpose(costVector) -- 1 x m
	
	local costPart1 = AqwamMatrixLibrary:dotProduct(transposedCostVector, kernelMatrix)
	
	costPart1 = AqwamMatrixLibrary:dotProduct(costPart1, kernelMatrix)
	
	costPart1 = AqwamMatrixLibrary:dotProduct(costPart1, costVector)
	
	costPart1 = 0.5 * costPart1
	
	local costPart2 = AqwamMatrixLibrary:dotProduct(transposedCostVector, kernelMatrix, labelVector) -- 1 x m, m x n
	
	local transposedLabelVector = AqwamMatrixLibrary:transpose(labelVector)
	
	local costPart3 = AqwamMatrixLibrary:dotProduct(transposedLabelVector, labelVector)
	
	costPart3 = 0.5 * costPart3
	
	local costPart4 = AqwamMatrixLibrary:dotProduct(transposedCostVector, kernelMatrix, costVector)
	
	costPart4 = (0.5 / cValue) * costPart4
	
	local cost = costPart1 - costPart2 + costPart3 + costPart4

	return cost

end

local function gradientDescent(modelParameters, individualkernelMatrix, labelVector, cValue)

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

		self:iterationWait()

		cost = calculateCost(self.ModelParameters, mappedFeatureMatrix, kernelMatrix, labelVector, self.cValue)

		self.ModelParameters = gradientDescent(self.ModelParameters, mappedFeatureMatrix, labelVector, self.cValue)
		
		numberOfIterations += 1
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)

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
