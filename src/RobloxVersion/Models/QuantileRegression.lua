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

local QuantileRegressionModel = {}

QuantileRegressionModel.__index = QuantileRegressionModel

setmetatable(QuantileRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultQuantile = 0.5

local function quantileLoss(hypothesisValue, labelValue, tau)
	
	local differenceValue = hypothesisValue - labelValue
	
	local multiplierValue = ((differenceValue < 0) and (tau - 1)) or tau
	
	local quantileLossValue = differenceValue * multiplierValue

	return quantileLossValue
	
end

function QuantileRegressionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = AqwamTensorLibrary:applyFunction(quantileLoss, hypothesisVector, labelVector, {self.QuantilesList}) 

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function QuantileRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function QuantileRegressionModel:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	local gradientWeightMatrix = AqwamTensorLibrary:applyFunction(function(lossValue, tau) return (lossValue < 0) and (tau - 1) or tau end, lossGradientMatrix, {self.QuantilesList})

	local lossFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), gradientWeightMatrix)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeMatrix = lossFunctionDerivativeMatrix end

	return lossFunctionDerivativeMatrix

end

function QuantileRegressionModel:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

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

function QuantileRegressionModel:update(lossGradientMatrix, clearAllMatrices)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local numberOfData = #lossGradientMatrix

	local lossFunctionDerivativeMatrix = self:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	self:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

	if (clearAllMatrices) then 
		
		self.featureMatrix = nil 
		
		self.lossFunctionDerivativeMatrix = nil
		
	end

end

function QuantileRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewQuantileRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewQuantileRegressionModel, QuantileRegressionModel)
	
	NewQuantileRegressionModel:setName("QuantileRegression")
	
	local QuantilesList = parameterDictionary.QuantilesList or {}
	
	if (#QuantilesList == 0) then QuantilesList[1] = defaultQuantile end

	NewQuantileRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewQuantileRegressionModel.QuantilesList = QuantilesList

	NewQuantileRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewQuantileRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewQuantileRegressionModel

end

function QuantileRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function QuantileRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function QuantileRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], #self.QuantilesList})

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

		local lossGradientMatrix = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		self:update(lossGradientMatrix, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function QuantileRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], #self.QuantilesList})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return QuantileRegressionModel
