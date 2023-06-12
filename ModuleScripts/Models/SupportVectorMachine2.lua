local BaseModel = require(script.Parent.BaseModel)

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCvalue = 0.3

local defaultTargetCost = 0

local defaultKernelFunction = "linear"

local kernelFunctionList = {
	
	["linear"] = function(x1, x2)
		
		local x2 = AqwamMatrixLibrary:transpose(x2)
		
		return AqwamMatrixLibrary:dotProduct(x1, x2)
		
	end,
	
	["polynomial"] = function(x1, x2, degree)
		
		local x2 = AqwamMatrixLibrary:transpose(x2)
		
		local dotProduct = AqwamMatrixLibrary:dotProduct(x1, x2)
		
		return math.pow(dotProduct, degree)
		
	end,
	
	["rbf"] = function(x1, x2, gamma)
		
		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		local part2 = AqwamMatrixLibrary:power(part1, 2)
		
		local sum = AqwamMatrixLibrary:sum(part2)
		
		local exponent = -gamma * sum
		
		return math.exp(exponent)
		
	end,
	
	["cosineSimilarity"] = function(x1, x2)
		
		local dotProduct = AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x2))
		
		local magnitude1 = math.sqrt(AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x1)))
		
		local magnitude2 = math.sqrt(AqwamMatrixLibrary:dotProduct(x2, AqwamMatrixLibrary:transpose(x2)))
		
		return dotProduct / (magnitude1 * magnitude2)
		
	end,

}

local function calculateKernel(x1, x2, kernelFunction, kernelParameters)
	
	if (kernelFunction == "linear") or (kernelFunction == "cosineSimilarity") then
		
		return kernelFunctionList[kernelFunction](x1, x2)
		
	elseif (kernelFunction == "polynomial") then
		
		local degree = kernelParameters.degree or 2
		
		return kernelFunctionList[kernelFunction](x1, x2, degree)
		
	elseif (kernelFunction == "rbf") then
		
		local gamma = kernelParameters.gamma or 1.0
		
		return kernelFunctionList[kernelFunction](x1, x2, gamma)
		
	end
	
end

local function calculateCost(modelParameters, featureMatrix, labelVector, cValue, kernelFunction, kernelParameters)
	
	local numberOfData = #featureMatrix
	
	local kernelMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)

	for i = 1, numberOfData do

		for j = 1, numberOfData do

			kernelMatrix[i][j] = calculateKernel({featureMatrix[i]}, {featureMatrix[j]}, kernelFunction, kernelParameters)

		end

	end
	
	local transposedModelMatrix = AqwamMatrixLibrary:transpose(modelParameters)

	local transposedLabelVector = AqwamMatrixLibrary:transpose(labelVector)
	
	local multipliedMatrix = AqwamMatrixLibrary:multiply(transposedLabelVector, kernelMatrix)
	
	local expandedTransposedModelMatrix = {}

	for i = 1, numberOfData do

		table.insert(expandedTransposedModelMatrix, transposedModelMatrix[1])

	end
	
	local hypothesisMatrix = AqwamMatrixLibrary:dotProduct(multipliedMatrix, expandedTransposedModelMatrix)
	
	local squaredDistanceMatrix = AqwamMatrixLibrary:power(hypothesisMatrix, 2)

	local sumSquaredDistance = AqwamMatrixLibrary:sum(squaredDistanceMatrix)
	
	local cost = (1 / (2 * numberOfData)) * sumSquaredDistance

	return cost
	
end

local function gradientDescent(modelParameters, featureMatrix, labelVector, cValue, kernelFunction, kernelParameters)
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = #featureMatrix[1]
	
	local kernelMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)
	
	for i = 1, numberOfData do
		
		for j = 1, numberOfData do
			
			kernelMatrix[i][j] = calculateKernel({featureMatrix[i]}, {featureMatrix[j]}, kernelFunction, kernelParameters)
			
		end
		
	end

	local transposedModelMatrix = AqwamMatrixLibrary:transpose(modelParameters)
	
	local transposedLabelVector = AqwamMatrixLibrary:transpose(labelVector)
	
	local multipliedMatrix = AqwamMatrixLibrary:multiply(transposedLabelVector, kernelMatrix)
	
	local expandedTransposedModelMatrix = {}
	
	for i = 1, numberOfData do
		
		table.insert(expandedTransposedModelMatrix, transposedModelMatrix[1])
		
	end
	
	local dotProductMatrix = AqwamMatrixLibrary:dotProduct(multipliedMatrix, expandedTransposedModelMatrix)

	local gradientMatrix = AqwamMatrixLibrary:subtract(dotProductMatrix, 1)
	
	local multipliedFeatureAndGradientMatrix = AqwamMatrixLibrary:multiply(gradientMatrix, featureMatrix)
	
	local costFunctionDerivatives = AqwamMatrixLibrary:verticalSum(multipliedFeatureAndGradientMatrix)
	
	costFunctionDerivatives = AqwamMatrixLibrary:divide(costFunctionDerivatives, numberOfData)
	
	costFunctionDerivatives = AqwamMatrixLibrary:transpose(costFunctionDerivatives)

	return costFunctionDerivatives

end

function SupportVectorMachineModel.new(maxNumberOfIterations, learningRate, cValue, targetCost, kernelFunction, kernelParameters)
	
	local NewSupportVectorMachine = BaseModel.new()
	
	setmetatable(NewSupportVectorMachine, SupportVectorMachineModel)
	
	NewSupportVectorMachine.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewSupportVectorMachine.learningRate = learningRate or defaultLearningRate
	
	NewSupportVectorMachine.cValue = cValue or defaultCvalue
	
	NewSupportVectorMachine.targetCost = targetCost or defaultTargetCost
	
	NewSupportVectorMachine.kernelFunction = kernelFunction or defaultKernelFunction
	
	NewSupportVectorMachine.kernelParameters = kernelParameters or {}
	
	NewSupportVectorMachine.validationFeatureMatrix = nil
	
	NewSupportVectorMachine.validationLabelVector = nil
	
	NewSupportVectorMachine.Optimizer = nil
	
	return NewSupportVectorMachine
end

function SupportVectorMachineModel:setOptimizer(Optimizer)
	self.Optimizer = Optimizer
end

function SupportVectorMachineModel:setParameters(maxNumberOfIterations, learningRate, cValue, targetCost, kernelFunction, kernelParameters)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.learningRate = learningRate or self.learningRate
	
	self.cValue = cValue or self.cValue
	
	self.targetCost = targetCost or self.targetCost
	
	self.kernelFunction = kernelFunction or self.kernelFunction
	
	self.kernelParameters = kernelParameters or self.kernelParameters
	
end

function SupportVectorMachineModel:setCValue(cValue)
	
	self.cValue = cValue or self.cValue
	
end

function SupportVectorMachineModel:train(featureMatrix, labelVector)
	
	local cost
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local costFunctionDerivatives
	
	local delta

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

	repeat
		
		numberOfIterations += 1
		
		costFunctionDerivatives = gradientDescent(self.ModelParameters, featureMatrix, labelVector, self.cValue, self.kernelFunction, self.kernelParameters)

		if (self.Optimizer) then
			
			costFunctionDerivatives = self.Optimizer:calculate(costFunctionDerivatives, delta)
			
		end

		delta = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivatives)
		
		self.ModelParameters = AqwamMatrixLibrary:add(self.ModelParameters, delta)

		cost = calculateCost(self.ModelParameters, featureMatrix, labelVector, self.cValue, self.kernelFunction, self.kernelParameters)
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)

	if (cost == math.huge) then
		
		warn("The model diverged! Please repeat the experiment or change the argument values.")
		
	end

	if self.Optimizer then
		
		self.Optimizer:reset()
		
	end

	return costArray
end

function SupportVectorMachineModel:predict(featureMatrix)
	
	local kernel = calculateKernel(featureMatrix, featureMatrix, self.kernelFunction, self.kernelParameters)
	
	local multiplied = AqwamMatrixLibrary:multiply(self.ModelParameters, kernel)
	
	local sum = AqwamMatrixLibrary:sum(multiplied)
	
	if (sum > 0) then
		
		return 1
		
	elseif (sum < 0) then
		
		return -1
		
	else
		
		return 0
		
	end 

end

return SupportVectorMachineModel
