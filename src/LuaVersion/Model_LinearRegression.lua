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

local LinearRegressionModel = {}

LinearRegressionModel.__index = LinearRegressionModel

setmetatable(LinearRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCostFunction = "L2"

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

function LinearRegressionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = lossFunctionList[self.costFunction](hypothesisVector, labelVector) 

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function LinearRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function LinearRegressionModel:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientMatrix)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeMatrix = lossFunctionDerivativeMatrix end

	return lossFunctionDerivativeMatrix

end

function LinearRegressionModel:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

	if (type(lossFunctionDerivativeMatrix) == "number") then lossFunctionDerivativeMatrix = {{lossFunctionDerivativeMatrix}} end
	
	local ModelParameters = self.ModelParameters
	
	local Regularizer = self.Regularizer
	
	local Optimizer = self.Optimizer
	
	local learningRate = self.learningRate

	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

		lossFunctionDerivativeMatrix = AqwamTensorLibrary:add(lossFunctionDerivativeMatrix, regularizationDerivatives)

	end

	lossFunctionDerivativeMatrix = AqwamTensorLibrary:divide(lossFunctionDerivativeMatrix, numberOfData)

	if (Optimizer) then 

		lossFunctionDerivativeMatrix = Optimizer:calculate(learningRate, lossFunctionDerivativeMatrix, ModelParameters) 

	else

		lossFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, lossFunctionDerivativeMatrix)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, lossFunctionDerivativeMatrix)

end

function LinearRegressionModel:update(lossGradientMatrix, clearAllMatrices)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local numberOfData = #lossGradientMatrix

	local lossFunctionDerivativeMatrix = self:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	self:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeMatrix = nil

	end

end

function LinearRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewLinearRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewLinearRegressionModel, LinearRegressionModel)
	
	NewLinearRegressionModel:setName("LinearRegression")

	NewLinearRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewLinearRegressionModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

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

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
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

		local lossGradientVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function LinearRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return LinearRegressionModel
