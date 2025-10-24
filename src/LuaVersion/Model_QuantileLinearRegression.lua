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

QuantileLinearRegressionModel = {}

QuantileLinearRegressionModel.__index = QuantileLinearRegressionModel

setmetatable(QuantileLinearRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local function quantileLoss(hypothesisValue, labelValue, tau)
	
	local differenceValue = hypothesisValue - labelValue
	
	local multiplierValue = ((differenceValue < 0) and (tau - 1)) or tau
	
	local quantileLossValue = differenceValue * multiplierValue

	return quantileLossValue
	
end

function QuantileLinearRegressionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = AqwamTensorLibrary:applyFunction(quantileLoss, hypothesisVector, labelVector, {self.quantilesList}) 

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function QuantileLinearRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function QuantileLinearRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	local gradientWeightMatrix = AqwamTensorLibrary:applyFunction(function(lossValue, tau) return (lossValue < 0) and (tau - 1) or tau end, lossMatrix, {self.quantilesList})

	local costFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), gradientWeightMatrix)

	if (self.areGradientsSaved) then self.costFunctionDerivativeMatrix = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix

end

function QuantileLinearRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

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

function QuantileLinearRegressionModel:update(lossMatrix, clearAllMatrices)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrix = self:calculateCostFunctionDerivativeMatrix(lossMatrix)

	self:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

	if (clearAllMatrices) then 
		
		self.featureMatrix = nil 
		
		self.costFunctionDerivativeMatrix = nil
		
	end

end

function QuantileLinearRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewQuantileLinearRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewQuantileLinearRegressionModel, QuantileLinearRegressionModel)
	
	NewQuantileLinearRegressionModel:setName("QuantileLinearRegression")
	
	local quantilesList = parameterDictionary.quantilesList or {}
	
	if (#quantilesList == 0) then quantilesList[1] = 0.5 end

	NewQuantileLinearRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewQuantileLinearRegressionModel.quantilesList = quantilesList

	NewQuantileLinearRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewQuantileLinearRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewQuantileLinearRegressionModel

end

function QuantileLinearRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function QuantileLinearRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function QuantileLinearRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], #self.quantilesList})

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

		local lossVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		self:update(lossVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function QuantileLinearRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], #self.quantilesList})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return QuantileLinearRegressionModel
