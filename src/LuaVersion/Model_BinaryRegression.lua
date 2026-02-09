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

local ZTableFunction = require("Core_ZTableFunction")

local BinaryRegressionModel = {}

BinaryRegressionModel.__index = BinaryRegressionModel

setmetatable(BinaryRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultBinaryFunction = "Logistic"

local defaultCostFunction = "BinaryCrossEntropy"

local function calculateProbabilityDensityFunctionValue(z)

	return (math.exp(-0.5 * math.pow(z, 2)) / math.sqrt(2 * math.pi))

end

local binaryFunctionList = {

	["Logistic"] = function (z) return (1/(1 + math.exp(-z))) end,
	
	["Logit"] = function (z) return (1/(1 + math.exp(-z))) end,
	
	["Probit"] = function(z) return ZTableFunction:getStandardNormalCumulativeDistributionFunctionValue(math.clamp(z, -3.9, 3.9)) end,

	["LogLog"] = function(z) return math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(z) return (1 - math.exp(-math.exp(z))) end,

	["Tanh"] = function (z) return math.tanh(z) end,
	
	["HardSigmoid"] = function (z)

		local x = (z + 1) / 2

		if (x < 0) then return 0 elseif (x > 1) then return 1 else return x end

	end,
	
	["SoftSign"] = function (z) return (z / (1 + math.abs(z))) end,
	
	["ArcTangent"] = function (z) return (2 / math.pi) * math.atan(z) end,

	["BipolarSigmoid"] = function (z) return (2 / (1 + math.exp(-z)) - 1) end,

}

local binaryFunctionGradientList = {
	
	["Logistic"] = function (h, z) return (h * (1 - h)) end,
	
	["Logit"] = function (h, z) return (h * (1 - h)) end,
	
	["Probit"] = function (h, z) return calculateProbabilityDensityFunctionValue(z) end,

	["LogLog"] = function(h, z) return -math.exp(z) * math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(h, z) return math.exp(z) * math.exp(-math.exp(z)) end,
	
	["Tanh"] = function (h, z) return (1 - math.pow(h, 2)) end,
	
	["HardSigmoid"] = function (h, z) return ((h <= 0 or h >= 1) and 0) or 0.5 end,
	
	["SoftSign"] = function (h, z) return (1 / ((1 + math.abs(z))^2)) end,
	
	["ArcTangent"] = function (h, z) return ((2 / math.pi) * (1 / (1 + z^2))) end,
	
	["BipolarSigmoid"] = function (h, z) 
		
		local sigmoidValue = 1 / (1 + math.exp(-z))
		
		return (2 * sigmoidValue * (1 - sigmoidValue))
		
	end,
	
}

local lossFunctionList = {
	
	["BinaryCrossEntropy"] = function (h, y) return -((y * math.log(h)) + ((1 - y) * math.log(1 - h))) end,
	
	["HingeLoss"] = function (h, y) return math.max(0, (1 - (h * y))) end,
	
	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,
	
	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,
	
}

local lossFunctionGradientList = {
	
	["BinaryCrossEntropy"] = function (h, y) return ((h - y) / (h * (1 - h))) end,
	
	["MeanSquaredError"] = function (h, y) return (2 * (h - y)) end,
	
	["MeanAbsoluteError"] = function (h, y) return math.sign(h - y) end,
	
	["HingeLoss"] = function (h, y)
		
		local scale = (((h * y) < 1) and 1) or 0
		
		return -(y * scale)
		
	end,
	
}

local minimumOutputValueList = {
	
	["0"] = {"Logistic", "Logit", "Probit", "LogLog", "ComplementaryLogLog", "HardSigmoid"}, -- 0.5 threshold for [0, 1] functions.

	["-1"] = {"Tanh", "SoftSign", "ArcTangent", "BipolarSigmoid"}, -- 0 threshold for [-1, 1] functions.
	
}

local function getCutOffValueList()
	
	local cutOffValueList = {}
	
	for stringMinimumOutputValue, binaryFunctionArray in pairs(minimumOutputValueList) do

		for _, binaryFunction in ipairs(binaryFunctionArray) do

			local minimumOutputValue = tonumber(stringMinimumOutputValue)

			cutOffValueList[binaryFunction] = (1 + minimumOutputValue) / 2

		end

	end
	
	return cutOffValueList
	
end

local cutOffValueList = getCutOffValueList()

function BinaryRegressionModel:calculateCost(hypothesisVector, labelVector)

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function BinaryRegressionModel:calculateHypothesisVector(featureMatrix, saveAllMatrices)

	local zVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	local hypothesisVector = AqwamTensorLibrary:applyFunction(binaryFunctionList[self.binaryFunction], zVector)
	
	if (saveAllMatrices) then 

		self.featureMatrix = featureMatrix

		self.zVector = zVector
		
		self.hypothesisVector = hypothesisVector

	end

	return hypothesisVector

end

function BinaryRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix
	
	local zVector = self.zVector
	
	local hypothesisVector = self.hypothesisVector

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	if (not zVector) then error("Z vector not found.") end
	
	if (not hypothesisVector) then error("Hypothesis vector not found.") end
	
	local binaryFunctionDerivativeVector = AqwamTensorLibrary:applyFunction(binaryFunctionGradientList[self.binaryFunction], hypothesisVector, zVector)
	
	binaryFunctionDerivativeVector = AqwamTensorLibrary:multiply(binaryFunctionDerivativeVector, lossGradientVector)
	
	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), binaryFunctionDerivativeVector)

	if (self.areGradientsSaved) then self.Gradients = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function BinaryRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end
	
	local ModelParameters = self.ModelParameters

	local Regularizer = self.Regularizer

	local Optimizer = self.Optimizer

	local learningRate = self.learningRate
	
	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

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

function BinaryRegressionModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)
	
	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.zVector = nil
		
		self.hypothesisVector = nil

	end

end

function BinaryRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewBinaryRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewBinaryRegressionModel, BinaryRegressionModel)
	
	NewBinaryRegressionModel:setName("BinaryRegression")

	NewBinaryRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewBinaryRegressionModel.binaryFunction = parameterDictionary.binaryFunction or defaultBinaryFunction
	
	NewBinaryRegressionModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewBinaryRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewBinaryRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewBinaryRegressionModel

end

function BinaryRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function BinaryRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function BinaryRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (numberOfFeatures ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local Optimizer = self.Optimizer
	
	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
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

		local lossGradientVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisVector, labelVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then
		
		if (cost == math.huge) then warn("The model diverged.") end
		
		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end
		
	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function BinaryRegressionModel:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1}) end

	local outputVector = self:calculateHypothesisVector(featureMatrix, false)

	if (returnOriginalOutput) then return outputVector end
	
	local binaryFunction = self.binaryFunction
	
	local minimumOutputValue = (minimumOutputValueList["0"][binaryFunction] and 0) or -1

	local cutOffValue = cutOffValueList[binaryFunction]
	
	local cutOffFunction = function(value) return ((value < cutOffValue) and minimumOutputValue) or 1 end

	local predictedLabelVector = AqwamTensorLibrary:applyFunction(cutOffFunction, outputVector)

	return predictedLabelVector, outputVector

end

return BinaryRegressionModel
