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

local defaultDegree = 3

local defaultSigma = 1

local mappingList = {

	["linear"] = function(X)

		return X

	end,

	["polynomial"] = function(X, degree)

		return AqwamMatrixLibrary:power(X, degree)

	end,

	["radialBasisFunction"] = function(X, sigma)
		
		local XSquaredVector = AqwamMatrixLibrary:power(X, 2)
		
		local sigmaSquaredVector = AqwamMatrixLibrary:power(sigma, 2)
		
		local multipliedSigmaSquaredVector = AqwamMatrixLibrary:multiply(-2, sigmaSquaredVector)
		
		local zVector = AqwamMatrixLibrary:divide(XSquaredVector, multipliedSigmaSquaredVector)
		
		return AqwamMatrixLibrary:applyFunction(math.exp, zVector)

	end,

	["cosineSimilarity"] = function(X)
		
		local XSquaredVector = AqwamMatrixLibrary:power(X, 2)

		local normXVector = AqwamMatrixLibrary:applyFunction(math.sqrt, XSquaredVector)
		
		return AqwamMatrixLibrary:divide(X, normXVector)

	end,

}

local hingeCostFunction = function (x) return math.max(0, x) end

local function calculateMapping(x, kernelFunction, kernelParameters)

	if (kernelFunction == "linear") or (kernelFunction == "cosineSimilarity") then

		return mappingList[kernelFunction](x)

	elseif (kernelFunction == "polynomial") then

		local degree = kernelParameters.degree or defaultDegree

		return mappingList[kernelFunction](x, degree)

	elseif (kernelFunction == "radialBasisFunction") then

		local sigma = kernelParameters.sigma or defaultSigma

		return mappingList[kernelFunction](x, sigma)

	end

end

local function calculateCost(modelParameters, kernelMatrix, labelVector, cValue)

	local cost

	local predictedValue

	local featureVector

	local mappedFeatureVector

	local regularizationTerm 

	local sumError

	local numberOfData = #labelVector

	local regularizationTermPart1 = AqwamMatrixLibrary:sum(AqwamMatrixLibrary:power(modelParameters, 2))

	local prediction = AqwamMatrixLibrary:dotProduct(kernelMatrix, modelParameters)

	local cost = AqwamMatrixLibrary:subtract(1, AqwamMatrixLibrary:multiply(labelVector, prediction))

	local hingeCostVector = AqwamMatrixLibrary:applyFunction(hingeCostFunction, cost)

	local hingeCostSum = AqwamMatrixLibrary:sum(hingeCostVector)

	cost = hingeCostSum/2

	regularizationTerm = (cValue / 2) * regularizationTermPart1

	cost += regularizationTerm

	return cost

end

local function gradientDescent(modelParameters, kernelMatrix, labelVector, cValue)
	
	local numberOfData = #labelVector

	local prediction = AqwamMatrixLibrary:dotProduct(kernelMatrix, modelParameters)

	local costPart1 = AqwamMatrixLibrary:multiply(labelVector, prediction)

	local cost = AqwamMatrixLibrary:subtract(1, costPart1)

	local hingeCost = AqwamMatrixLibrary:applyFunction(hingeCostFunction, cost)

	local hingeCostDerivatives = AqwamMatrixLibrary:createMatrix(#kernelMatrix, #kernelMatrix[1])

	for i = 1, numberOfData, 1 do

		if (hingeCost[i][1] ~= 0) then

			hingeCostDerivatives = AqwamMatrixLibrary:multiply(-cValue, labelVector, kernelMatrix)

		end

	end

	local costFunctionDerivativesPart1 = AqwamMatrixLibrary:transpose(AqwamMatrixLibrary:verticalSum(hingeCostDerivatives))

	local regularizationPart1 = AqwamMatrixLibrary:divide(modelParameters, cValue)

	local costFunctionDerivatives = AqwamMatrixLibrary:add(regularizationPart1, costFunctionDerivativesPart1)

	costFunctionDerivatives = AqwamMatrixLibrary:divide(costFunctionDerivatives, numberOfData)

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

	local kernelMatrix  = calculateMapping(featureMatrix, self.kernelFunction, self.kernelParameters)

	repeat

		self:iterationWait()

		numberOfIterations += 1

		costFunctionDerivatives = gradientDescent(self.ModelParameters, kernelMatrix, labelVector, self.cValue)

		if (self.Optimizer) then

			costFunctionDerivatives = self.Optimizer:calculate(self.learningRate, costFunctionDerivatives)

		else

			costFunctionDerivatives = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivatives)

		end

		self.ModelParameters = AqwamMatrixLibrary:subtract(self.ModelParameters, costFunctionDerivatives)

		cost = calculateCost(self.ModelParameters, kernelMatrix, labelVector, self.cValue, self.kernelFunction, self.kernelParameters)

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)

	if (cost == math.huge) then

		warn("The model diverged! Please repeat the experiment or change the argument values.")

	end

	if (self.Optimizer) and (self.AutoResetOptimizers) then self.Optimizer:reset() end

	return costArray
	
end

function SupportVectorMachineModel:predict(featureMatrix, returnOriginalOutput)

	local mappedFeatureVector = calculateMapping(featureMatrix, self.kernelFunction, self.kernelParameters)

	local originalPredictedVector = AqwamMatrixLibrary:dotProduct(mappedFeatureVector, self.ModelParameters)
	
	if (returnOriginalOutput == true) then return originalPredictedVector end
	
	local clampFunction = function (x) return math.clamp(x, -1, 1) end
	
	local predictedVector = AqwamMatrixLibrary:applyFunction(clampFunction, originalPredictedVector)
	
	return predictedVector

end

return SupportVectorMachineModel
