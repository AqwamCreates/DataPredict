--[[

	--------------------------------------------------------------------

	Aqwam's Regression, Deep And Reinforcement Learning Library (DataPredict)

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

local SupportVectorRegressionGradientVariantModel = {}

SupportVectorRegressionGradientVariantModel.__index = SupportVectorRegressionGradientVariantModel

setmetatable(SupportVectorRegressionGradientVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCValue = 1

local defaultEpsilon = 1

function SupportVectorRegressionGradientVariantModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local epsilon = self.epsilon
	
	local errorVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)
	
	local positiveSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, errorValue - epsilon) end, errorVector)

	local negativeSlackVariableVector = AqwamTensorLibrary:applyFunction(function(errorValue) return math.max(0, -errorValue - epsilon) end, errorVector)

	local slackVariableVector = AqwamTensorLibrary:add(positiveSlackVariableVector, negativeSlackVariableVector)

	local totalCost = AqwamTensorLibrary:sum(slackVariableVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = (self.cValue * totalCost) / #labelVector

	return averageCost

end

function SupportVectorRegressionGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorRegressionGradientVariantModel:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientMatrix)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeMatrix = lossFunctionDerivativeMatrix end

	return lossFunctionDerivativeMatrix

end

function SupportVectorRegressionGradientVariantModel:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

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

function SupportVectorRegressionGradientVariantModel:update(lossGradientMatrix, clearAllMatrices)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local numberOfData = #lossGradientMatrix

	local lossFunctionDerivativeMatrix = self:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	self:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeMatrix = nil

	end

end

function SupportVectorRegressionGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorRegressionGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorRegressionGradientVariantModel, SupportVectorRegressionGradientVariantModel)
	
	NewSupportVectorRegressionGradientVariantModel:setName("SupportVectorRegressionGradientVariant")

	NewSupportVectorRegressionGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorRegressionGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue
	
	NewSupportVectorRegressionGradientVariantModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewSupportVectorRegressionGradientVariantModel.Optimizer = parameterDictionary.Optimizer

	NewSupportVectorRegressionGradientVariantModel.Regularizer = parameterDictionary.Regularizer

	return NewSupportVectorRegressionGradientVariantModel

end

function SupportVectorRegressionGradientVariantModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function SupportVectorRegressionGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorRegressionGradientVariantModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue
	
	local epsilon = self.epsilon

	local Optimizer = self.Optimizer
	
	local functionToApply = function(errorValue) return ((errorValue > epsilon) and (errorValue - epsilon)) or ((errorValue < -epsilon) and (errorValue + epsilon)) or 0 end

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
		
		local errorVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		local lossGradientVector = AqwamTensorLibrary:applyFunction(functionToApply, errorVector)
		
		lossGradientVector = AqwamTensorLibrary:multiply(cValue, lossGradientVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function SupportVectorRegressionGradientVariantModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return SupportVectorRegressionGradientVariantModel
