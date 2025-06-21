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

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

LogisticRegressionModel = {}

LogisticRegressionModel.__index = LogisticRegressionModel

setmetatable(LogisticRegressionModel, GradientMethodBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "Sigmoid"

local sigmoidFunctionList = {

	["Sigmoid"] = function (z) return 1/(1 + math.exp(-1 * z)) end,

	["Tanh"] = function (z) return math.tanh(z) end

}

local lossFunctionList = {

	["Sigmoid"] = function (h, y) return -(y * math.log(h) + (1 - y) * math.log(1 - h)) end,

	["Tanh"] = function (h, y) return ((h - y)^2) / 2 end

}

local derivativeLossFunctionList = {

	["Sigmoid"] = function (h, y) return (h - y) end,

	["Tanh"] = function (h, y) return (h - y) * (1 - math.pow(h, 2)) end

}

local cutOffFunctionList = {

	["Sigmoid"] = function (x) 

		if (x >= 0.5) then 

			return 1

		else 

			return 0 

		end 

	end,

	["Tanh"] = function (x) 

		if (x > 0) then 

			return 1

		elseif (x < 0) then

			return -1

		else

			return 0

		end 

	end

}

function LogisticRegressionModel:calculateCost(hypothesisVector, labelVector, numberOfData)

	local costVector = AqwamMatrixLibrary:applyFunction(lossFunctionList[self.sigmoidFunction], hypothesisVector, labelVector)

	local totalCost = AqwamMatrixLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateRegularization(self.ModelParameters) end

	local averageCost = totalCost / numberOfData

	return averageCost

end

function LogisticRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local zVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then 

		self.featureMatrix = featureMatrix

	end

	if (type(zVector) == "number") then zVector = {{zVector}} end

	local hypothesisVector = AqwamMatrixLibrary:applyFunction(sigmoidFunctionList[self.sigmoidFunction], zVector)

	return hypothesisVector

end

function LogisticRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local featureMatrix = self.featureMatrix

	if (featureMatrix == nil) then error("Feature matrix not found.") end

	local costFunctionDerivativeMatrix = AqwamMatrixLibrary:dotProduct(AqwamMatrixLibrary:transpose(featureMatrix), lossMatrix)

	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix

end

function LogisticRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

	if (type(costFunctionDerivativeMatrix) == "number") then costFunctionDerivativeMatrix = {{costFunctionDerivativeMatrix}} end

	if (self.Regularizer) then

		local regularizationDerivatives = self.Regularizer:calculateRegularizationDerivatives(self.ModelParameters)

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:add(costFunctionDerivativeMatrix, regularizationDerivatives)

	end

	costFunctionDerivativeMatrix = AqwamMatrixLibrary:divide(costFunctionDerivativeMatrix, numberOfData)

	if (self.Optimizer) then

		costFunctionDerivativeMatrix = self.Optimizer:calculate(self.learningRate, costFunctionDerivativeMatrix) 

	else

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivativeMatrix)

	end

	self.ModelParameters = AqwamMatrixLibrary:subtract(self.ModelParameters, costFunctionDerivativeMatrix)

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

	local cost

	local costArray = {}

	local numberOfIterations = 0

	local numberOfData = #featureMatrix

	local derivativeLossFunctionToApply = derivativeLossFunctionList[self.sigmoidFunction] 

	local Regularizer = self.Regularizer

	local maximumNumberOfIterations = self.maximumNumberOfIterations

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end

	if (self.ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local hypothesisVector = self:calculateHypothesisVector(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisVector, labelVector, numberOfData)

		end)

		if cost then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossVector = AqwamMatrixLibrary:applyFunction(derivativeLossFunctionToApply, hypothesisVector, labelVector)

		self:update(lossVector, true, false)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	if (self.Optimizer) and (self.autoResetOptimizers) then self.Optimizer:reset() end

	return costArray

end

function LogisticRegressionModel:predict(featureMatrix, returnOriginalOutput)

	local outputVector = self:calculateHypothesisVector(featureMatrix, false)

	if (returnOriginalOutput) then return outputVector end

	local cutOffFunction = cutOffFunctionList[self.sigmoidFunction]

	local predictedLabelVector = AqwamMatrixLibrary:applyFunction(cutOffFunction, outputVector)

	return predictedLabelVector, outputVector

end

return LogisticRegressionModel