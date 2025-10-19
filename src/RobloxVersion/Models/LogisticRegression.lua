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

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

LogisticRegressionModel = {}

LogisticRegressionModel.__index = LogisticRegressionModel

setmetatable(LogisticRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "Sigmoid"

local sigmoidFunctionList = {

	["Sigmoid"] = function (z) return 1/(1 + math.exp(-1 * z)) end,

	["Tanh"] = function (z) return math.tanh(z) end,
	
	["Softsign"] = function (z) return z / (1 + math.abs(z)) end,

	["HardSigmoid"] = function (z)
		
		local x = (z + 1) / 2
		
		if x < 0 then return 0 elseif x > 1 then return 1 else return x end
		
	end,

	["Swish"] = function (z) return z / (1 + math.exp(-z)) end,

	["BipolarSigmoid"] = function (z) return 2 / (1 + math.exp(-z)) - 1 end,

	["GELU"] = function (z) return 0.5 * z * (1 + math.erf(z / math.sqrt(2))) end,

	["Arctangent"] = function (z) return (2 / math.pi) * math.atan(z) end

}

local derivativeLossFunctionList = {

	["Sigmoid"] = function (h, y) return (h - y) end,

	["Tanh"] = function (h, y) return (h - y) * (1 - math.pow(h, 2)) end,
	
	["HardSigmoid"] = function (h, y)
		local grad
		if h <= 0 or h >= 1 then
			grad = 0
		else
			grad = 0.5
		end
		return (h - y) * grad
	end,

	["Softsign"] = function (h, y) return (h - y) *  1 / ((1 + math.abs(h))^2) end,

	["ArcTangent"] = function (h, y) return (h - y) * (2 / math.pi) * (1 / (1 + h^2)) end,

	["Swish"] = function (h, y)
		
		local sigmoid_h = 1 / (1 + math.exp(-h))
		
		return (h - y) * sigmoid_h + h * sigmoid_h * (1 - sigmoid_h)
		
	end,

	["BipolarSigmoid"] = function (h, y) return (h - y) * 0.5 * (1 - h^2) end,


}

local lossFunctionList = {

	["Sigmoid"] = function (h, y) return -(y * math.log(h) + (1 - y) * math.log(1 - h)) end,

	["Tanh"] = function (h, y) return ((h - y)^2) / 2 end,
	
	["HardSigmoid"] = function (h, y) return -(y * math.log(h + 1e-10) + (1 - y) * math.log(1 - h + 1e-10)) end,

	["Softsign"] = function (h, y) return ((h - y)^2) / 2 end,

	["ArcTangent"] = function (h, y) return ((h - y)^2) / 2 end,

	["Swish"] = function (h, y) return ((h - y)^2) / 2 end,

	["BipolarSigmoid"] = function (h, y) return ((h - y)^2) / 2 end,

}

local cutOffList = {
	
	["0.5"] = {"Sigmoid", "HardSigmoid", "Swish"}, -- 0.5 threshold for [0,1] functions

	["0"] = {"Tanh", "Softsign", "ArcTangent", "BipolarSigmoid"}, -- 0 threshold for [-1,1] functions
	
}

local function getCutOffFunction(sigmoidFunction)
	
	for stringCutOffValue, sigmoidFunctionArray in pairs(cutOffList) do

		if (table.find(sigmoidFunctionArray, sigmoidFunction)) then

			local cutOffValue = tonumber(stringCutOffValue)

			local negativeValue = (cutOffValue == 0.5) and 0 or -1

			local cutOffFunction = function(x) 

				if (x > cutOffValue) then return 1 end

				if (x < cutOffValue) then return negativeValue end

				return 0

			end

			return cutOffFunction

		end

	end
	
	error("Cut-off function not found for " .. tostring(sigmoidFunction) .. ".")
	
end

function LogisticRegressionModel:calculateCost(hypothesisVector, labelVector)

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.sigmoidFunction], hypothesisVector, labelVector)

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

function LogisticRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local costFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossMatrix)

	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix

end

function LogisticRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

	if (type(costFunctionDerivativeMatrix) == "number") then costFunctionDerivativeMatrix = {{costFunctionDerivativeMatrix}} end
	
	local ModelParameters = self.ModelParameters

	local Regularizer = self.Regularizer

	local Optimizer = self.Optimizer

	local learningRate = self.learningRate
	
	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:add(costFunctionDerivativeMatrix, regularizationDerivatives)

	end

	costFunctionDerivativeMatrix = AqwamTensorLibrary:divide(costFunctionDerivativeMatrix, numberOfData)

	if (Optimizer) then

		costFunctionDerivativeMatrix = Optimizer:calculate(learningRate, costFunctionDerivativeMatrix, ModelParameters) 

	else

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrix)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, costFunctionDerivativeMatrix)

end

function LogisticRegressionModel:update(lossMatrix, clearFeatureMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrix = self:calculateCostFunctionDerivativeMatrix(lossMatrix)

	self:gradientDescent(costFunctionDerivativeMatrix, numberOfData)
	
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

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters!") end

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

		local lossVector = AqwamTensorLibrary:applyFunction(derivativeLossFunctionToApply, hypothesisVector, labelVector)

		self:update(lossVector, true, false)

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
