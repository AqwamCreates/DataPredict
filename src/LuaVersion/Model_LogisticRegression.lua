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

local LogisticRegressionModel = {}

LogisticRegressionModel.__index = LogisticRegressionModel

setmetatable(LogisticRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "Sigmoid"

local sigmoidFunctionList = {

	["Sigmoid"] = function (z) return 1/(1 + math.exp(-1 * z)) end,

	["Tanh"] = function (z) return math.tanh(z) end,
	
	["HardSigmoid"] = function (z)

		local x = (z + 1) / 2

		if (x < 0) then return 0 elseif (x > 1) then return 1 else return x end

	end,
	
	["SoftSign"] = function (z) return z / (1 + math.abs(z)) end,
	
	["ArcTangent"] = function (z) return (2 / math.pi) * math.atan(z) end,

	["Swish"] = function (z) return z / (1 + math.exp(-z)) end,

	["BipolarSigmoid"] = function (z) return 2 / (1 + math.exp(-z)) - 1 end,

}

local derivativeLossFunctionList = {

	["Sigmoid"] = function (h, y) return (h - y) end,

	["Tanh"] = function (h, y) return (h - y) * (1 - math.pow(h, 2)) end,
	
	["HardSigmoid"] = function (h, y) return (h - y) * ((h <= 0 or h >= 1) and 0) or 0.5 end,

	["SoftSign"] = function (h, y) return (h - y) *  1 / ((1 + math.abs(h))^2) end,
	
	["ArcTangent"] = function (h, y) return (h - y) * (2 / math.pi) * (1 / (1 + h^2)) end,

	["Swish"] = function (h, y)
		
		local sigmoidValue = 1 / (1 + math.exp(-h))
		
		return (h - y) * sigmoidValue + h * sigmoidValue * (1 - sigmoidValue)
		
	end,

	["BipolarSigmoid"] = function (h, y) return (h - y) * 0.5 * (1 - h^2) end,

}

local lossFunctionGradientList = {

	["Sigmoid"] = function (h, y) return -(y * math.log(h) + (1 - y) * math.log(1 - h)) end,

	["Tanh"] = function (h, y) return ((h - y)^2) / 2 end,
	
	["HardSigmoid"] = function (h, y) return -(y * math.log(h) + (1 - y) * math.log(1 - h)) end,

	["SoftSign"] = function (h, y) return ((h - y)^2) / 2 end,

	["ArcTangent"] = function (h, y) return ((h - y)^2) / 2 end,

	["Swish"] = function (h, y) return ((h - y)^2) / 2 end,

	["BipolarSigmoid"] = function (h, y) return ((h - y)^2) / 2 end,

}

local cutOffList = {
	
	["0.5"] = {"Sigmoid", "HardSigmoid", "Swish"}, -- 0.5 threshold for [0, 1] functions.

	["0"] = {"Tanh", "SoftSign", "ArcTangent", "BipolarSigmoid"}, -- 0 threshold for [-1, 1] functions.
	
}

local function getCutOffFunction(sigmoidFunction)
	
	for stringCutOffValue, sigmoidFunctionArray in pairs(cutOffList) do

		if (table.find(sigmoidFunctionArray, sigmoidFunction)) then

			local cutOffValue = tonumber(stringCutOffValue)

			local lowerValue = (cutOffValue == 0.5) and 0 or -1

			local cutOffFunction = function(x) 

				if (x > cutOffValue) then return 1 end

				if (x < cutOffValue) then return lowerValue end

				return cutOffValue

			end

			return cutOffFunction

		end

	end
	
	error("Unknown cut-off function.")
	
end

function LogisticRegressionModel:calculateCost(hypothesisVector, labelVector)

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientList[self.sigmoidFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function LogisticRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local zVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then 

		self.featureMatrix = featureMatrix

	end

	local hypothesisVector = AqwamTensorLibrary:applyFunction(sigmoidFunctionList[self.sigmoidFunction], zVector)

	return hypothesisVector

end

function LogisticRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientVector)

	if (self.areGradientsSaved) then self.Gradients = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function LogisticRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

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

function LogisticRegressionModel:update(lossGradientVector, clearFeatureMatrix)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)
	
	if (clearFeatureMatrix) then self.featureMatrix = nil end

end

function LogisticRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewLogisticRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewLogisticRegressionModel, LogisticRegressionModel)
	
	NewLogisticRegressionModel:setName("LogisticRegression")

	NewLogisticRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewLogisticRegressionModel.sigmoidFunction = parameterDictionary.sigmoidFunction or defaultSigmoidFunction

	NewLogisticRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewLogisticRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewLogisticRegressionModel

end

function LogisticRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function LogisticRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function LogisticRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local derivativeLossFunctionToApply = derivativeLossFunctionList[self.sigmoidFunction] 
	
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

		local lossGradientVector = AqwamTensorLibrary:applyFunction(derivativeLossFunctionToApply, hypothesisVector, labelVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then
		
		if (cost == math.huge) then warn("The model diverged.") end
		
		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end
		
	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function LogisticRegressionModel:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1}) end

	local outputVector = self:calculateHypothesisVector(featureMatrix, false)

	if (returnOriginalOutput) then return outputVector end
	
	local cutOffFunction = getCutOffFunction(self.sigmoidFunction)

	local predictedLabelVector = AqwamTensorLibrary:applyFunction(cutOffFunction, outputVector)

	return predictedLabelVector, outputVector

end

return LogisticRegressionModel
