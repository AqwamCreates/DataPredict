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

local GradientMethodBaseModel = require("Model_GradientMethodBaseModel")

LinearRegressionModel = {}

LinearRegressionModel.__index = LinearRegressionModel

setmetatable(LinearRegressionModel, GradientMethodBaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultLossFunction = "L2"

local lossFunctionList = {

	["L1"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		return AqwamTensorLibrary:applyFunction(math.abs, part1) 

	end,

	["L2"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		local part2 = AqwamTensorLibrary:power(part1, 2) 

		return AqwamTensorLibrary:divide(part2, 2)

	end,

}

function LinearRegressionModel:calculateCost(hypothesisVector, labelVector, numberOfData)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = lossFunctionList[self.lossFunction](hypothesisVector, labelVector) 

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateRegularization(self.ModelParameters) end

	local averageCost = totalCost / numberOfData

	return averageCost

end

function LinearRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then 

		self.featureMatrix = featureMatrix

	end

	return hypothesisVector

end

function LinearRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local featureMatrix = self.featureMatrix

	if (featureMatrix == nil) then error("Feature matrix not found.") end

	local costFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossMatrix)

	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix

end

function LinearRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

	if (type(costFunctionDerivativeMatrix) == "number") then costFunctionDerivativeMatrix = {{costFunctionDerivativeMatrix}} end

	if (self.Regularizer) then

		local regularizationDerivatives = self.Regularizer:calculateRegularizationDerivatives(self.ModelParameters)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:add(costFunctionDerivativeMatrix, regularizationDerivatives)

	end

	costFunctionDerivativeMatrix = AqwamTensorLibrary:divide(costFunctionDerivativeMatrix, numberOfData)

	if (self.Optimizer) then 

		costFunctionDerivativeMatrix = self.Optimizer:calculate(self.learningRate, costFunctionDerivativeMatrix) 

	else

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(self.learningRate, costFunctionDerivativeMatrix)

	end

	local newModelParameters = AqwamTensorLibrary:subtract(self.ModelParameters, costFunctionDerivativeMatrix)

	return newModelParameters

end

function LinearRegressionModel:update(lossMatrix, clearFeatureMatrix, doNotUpdateModelParameters)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrix = self:calculateCostFunctionDerivativeMatrix(lossMatrix)

	self.ModelParameters = self:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

	if (clearFeatureMatrix) then self.featureMatrix = nil end

end

function LinearRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewLinearRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewLinearRegressionModel, LinearRegressionModel)
	
	NewLinearRegressionModel:setName("LinearRegression")

	NewLinearRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewLinearRegressionModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction

	NewLinearRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewLinearRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewLinearRegressionModel

end

function LinearRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function LinearRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function LinearRegressionModel:train(featureMatrix, labelVector)

	local cost

	local costArray = {}

	local numberOfIterations = 0

	local numberOfData = #featureMatrix

	local lossFunction = self.lossFunction

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

		local lossVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		self:update(lossVector, true, false)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values") end

	if (self.Optimizer) and (self.autoResetOptimizers) then self.Optimizer:reset() end

	return costArray

end

function LinearRegressionModel:predict(featureMatrix)

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (type(predictedVector) == "number") then predictedVector = {{predictedVector}} end

	return predictedVector

end

return LinearRegressionModel