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

PoissonLinearRegressionModel = {}

PoissonLinearRegressionModel.__index = PoissonLinearRegressionModel

setmetatable(PoissonLinearRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

function PoissonLinearRegressionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local logPredictedCountVector = AqwamTensorLibrary:applyFunction(math.log, hypothesisVector)
	
	local observedTimesLogPredictedVector = AqwamTensorLibrary:multiply(labelVector, logPredictedCountVector)
	
	local costVector = AqwamTensorLibrary:subtract(hypothesisVector, observedTimesLogPredictedVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function PoissonLinearRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)
	
	local exponentTermVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	local hypothesisVector = AqwamTensorLibrary:applyFunction(math.exp, exponentTermVector)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function PoissonLinearRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local costFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossMatrix)

	if (self.areGradientsSaved) then self.costFunctionDerivativeMatrix = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix

end

function PoissonLinearRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

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

function PoissonLinearRegressionModel:update(lossMatrix, clearAllMatrices)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrix = self:calculateCostFunctionDerivativeMatrix(lossMatrix)

	self:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.costFunctionDerivativeMatrix = nil

	end

end

function PoissonLinearRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewPoissonLinearRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewPoissonLinearRegressionModel, PoissonLinearRegressionModel)
	
	NewPoissonLinearRegressionModel:setName("PoissonLinearRegression")

	NewPoissonLinearRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewPoissonLinearRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewPoissonLinearRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewPoissonLinearRegressionModel

end

function PoissonLinearRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function PoissonLinearRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function PoissonLinearRegressionModel:train(featureMatrix, labelVector)

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

function PoissonLinearRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local exponentTermVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	local predictedVector = AqwamTensorLibrary:applyFunction(math.exp, exponentTermVector)

	return predictedVector

end

return PoissonLinearRegressionModel
