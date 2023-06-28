local BaseModel = require(script.Parent.BaseModel)

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCvalue = 1

local defaultTargetCost = 0

local defaultKernelFunction = "linear"

local defaultDegree = 3

local defaultGamma = 1

local kernelFunctionList = {

	["linear"] = function(x1)

		local x2 = AqwamMatrixLibrary:transpose(x1)

		return AqwamMatrixLibrary:dotProduct(x1, x2)

	end,

	["polynomial"] = function(x1, degree)

		local x2 = AqwamMatrixLibrary:transpose(x1)

		local dotProduct = AqwamMatrixLibrary:dotProduct(x1, x2)

		return AqwamMatrixLibrary:power(dotProduct, degree)

	end,

	["rbf"] = function(x1, gamma)

		local numberOfRows = #x1

		local pairwiseDistances = {}

		for i = 1, numberOfRows, 1 do

			pairwiseDistances[i] = {}

			local x2 = {x1[i]}

			for j = 1, numberOfRows, 1 do

				local x3 = {x1[j]}

				local distance = AqwamMatrixLibrary:subtract(x2, x3)

				local squaredDistance = AqwamMatrixLibrary:power(distance, 2)

				local sumSquaredDistance = AqwamMatrixLibrary:sum(squaredDistance)

				pairwiseDistances[i][j] = math.sqrt(sumSquaredDistance)

			end

		end

		local squaredDistances = AqwamMatrixLibrary:power(pairwiseDistances, 2)

		local exponent = AqwamMatrixLibrary:multiply(-gamma, squaredDistances)

		local k = AqwamMatrixLibrary:applyFunction(math.exp, exponent)

		return k

	end,

	["cosineSimilarity"] = function(x1)

		local dotProductMatrix = AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x1))

		local magnitudeMatrix = AqwamMatrixLibrary:applyFunction(math.sqrt, AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x1)))

		local multiplyMatrix = AqwamMatrixLibrary:multiply(magnitudeMatrix, magnitudeMatrix)

		return AqwamMatrixLibrary:divide(dotProductMatrix, multiplyMatrix)

	end,

}

local mappingList = {

	["linear"] = function(x)

		return x

	end,

	["polynomial"] = function(x, degree)

		return AqwamMatrixLibrary:power(x, degree)

	end,

	["rbf"] = function(x, gamma)

		local multiplied = AqwamMatrixLibrary:multiply(x, -gamma)

		return AqwamMatrixLibrary:applyFunction(math.exp, multiplied)

	end,

	["cosineSimilarity"] = function(x)

		return x

	end,

}

local function calculateKernel(x, kernelFunction, kernelParameters)

	if (kernelFunction == "linear") or (kernelFunction == "cosineSimilarity") then

		return kernelFunctionList[kernelFunction](x)

	elseif (kernelFunction == "polynomial") then

		local degree = kernelParameters.degree or defaultDegree

		return kernelFunctionList[kernelFunction](x, degree)

	elseif (kernelFunction == "rbf") then

		local gamma = kernelParameters.gamma or defaultGamma

		return kernelFunctionList[kernelFunction](x, gamma)

	end

end

local function calculateMapping(x, kernelFunction, kernelParameters)

	if (kernelFunction == "linear") or (kernelFunction == "cosineSimilarity") then

		return mappingList[kernelFunction](x)

	elseif (kernelFunction == "polynomial") then

		local degree = kernelParameters.degree or defaultDegree

		return mappingList[kernelFunction](x, degree)

	elseif (kernelFunction == "rbf") then

		local gamma = kernelParameters.gamma or defaultGamma

		return mappingList[kernelFunction](x, gamma)

	end

end

local function calculateCost(modelParameters, featureMatrix, labelVector, cValue, kernelFunction, kernelParameters)

	local cost

	local predictedValue

	local featureVector

	local mappedFeatureVector

	local regularizationTerm 

	local sumError

	local numberOfData = #featureMatrix

	local squaredErrorVector = AqwamMatrixLibrary:createMatrix(numberOfData, 1)

	local dotProductedModelParameters = AqwamMatrixLibrary:dotProduct(AqwamMatrixLibrary:transpose(modelParameters), modelParameters)

	local divisionConstant = (1 / 2)

	for i = 1, numberOfData, 1 do

		featureVector = {featureMatrix[i]}

		mappedFeatureVector = calculateMapping(featureVector, kernelFunction, kernelParameters)

		predictedValue = AqwamMatrixLibrary:dotProduct(mappedFeatureVector, modelParameters)

		squaredErrorVector[i][1] = (predictedValue - labelVector[i][1])^2

	end

	sumError = AqwamMatrixLibrary:sum(squaredErrorVector)

	cost = divisionConstant * sumError

	regularizationTerm = cValue * divisionConstant * dotProductedModelParameters

	cost += regularizationTerm

	return cost

end

local function gradientDescent(modelParameters, kernelMatrix, featureMatrix, labelVector, cValue)

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local transposedModelMatrix = AqwamMatrixLibrary:transpose(modelParameters)

	local transposedLabelVector = AqwamMatrixLibrary:transpose(labelVector)

	local multipliedMatrix = AqwamMatrixLibrary:subtract(kernelMatrix, transposedLabelVector)

	local expandedTransposedModelMatrix = {}

	for i = 1, numberOfData do

		table.insert(expandedTransposedModelMatrix, transposedModelMatrix[1])

	end

	local dotProductMatrix = AqwamMatrixLibrary:dotProduct(multipliedMatrix, expandedTransposedModelMatrix)

	local gradientMatrix = AqwamMatrixLibrary:subtract(dotProductMatrix, 1)

	local multipliedFeatureAndGradientMatrix = AqwamMatrixLibrary:multiply(gradientMatrix, featureMatrix)

	local NewModelParameters = AqwamMatrixLibrary:verticalSum(multipliedFeatureAndGradientMatrix)

	NewModelParameters = AqwamMatrixLibrary:divide(NewModelParameters, cValue)

	NewModelParameters = AqwamMatrixLibrary:transpose(NewModelParameters)

	NewModelParameters = AqwamMatrixLibrary:multiply(-1, NewModelParameters)

	return NewModelParameters

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

	local previousCostFunctionDerivatives
	
	local kernelMatrix  = calculateKernel(featureMatrix, self.kernelFunction, self.kernelParameters)

	repeat

		self:iterationWait()

		numberOfIterations += 1

		costFunctionDerivatives = gradientDescent(self.ModelParameters, kernelMatrix, featureMatrix, labelVector, self.cValue)

		if (self.Optimizer) then

			costFunctionDerivatives = self.Optimizer:calculate(costFunctionDerivatives, previousCostFunctionDerivatives)

		end

		costFunctionDerivatives = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivatives)

		previousCostFunctionDerivatives = costFunctionDerivatives

		self.ModelParameters = costFunctionDerivatives

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

	local mappedFeatureVector = calculateMapping(featureMatrix, self.kernelFunction, self.kernelParameters)

	local predictedValue = AqwamMatrixLibrary:dotProduct(mappedFeatureVector, self.ModelParameters)

	if (predictedValue > 0) then

		return 1

	elseif (predictedValue < 0) then

		return -1

	else

		return 0

	end

end

return SupportVectorMachineModel
