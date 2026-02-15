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

local LinearRegressionCovariancePreconditionedVariantModel = {}

LinearRegressionCovariancePreconditionedVariantModel.__index = LinearRegressionCovariancePreconditionedVariantModel

setmetatable(LinearRegressionCovariancePreconditionedVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCostFunction = "MeanSquaredError"

local lossFunctionList = {

	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,

	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,

}

local lossFunctionGradientList = {

	["MeanSquaredError"] = function (h, y) return (2 * (h - y)) end,

	["MeanAbsoluteError"] = function (h, y) return math.sign(h - y) end,

}

local function calculatePMatrix(featureMatrix)

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local pMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	pMatrix = AqwamTensorLibrary:inverse(pMatrix)

	pMatrix = AqwamTensorLibrary:dotProduct(pMatrix, transposedFeatureMatrix)

	return pMatrix

end

function LinearRegressionCovariancePreconditionedVariantModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function LinearRegressionCovariancePreconditionedVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function LinearRegressionCovariancePreconditionedVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local pMatrix = self.pMatrix or calculatePMatrix(featureMatrix)

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(pMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function LinearRegressionCovariancePreconditionedVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end
	
	local ModelParameters = self.ModelParameters
	
	local Regularizer = self.Regularizer
	
	local learningRate = self.learningRate

	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

		lossFunctionDerivativeVector = AqwamTensorLibrary:add(lossFunctionDerivativeVector, regularizationDerivatives)

	end

	lossFunctionDerivativeVector = AqwamTensorLibrary:divide(lossFunctionDerivativeVector, numberOfData)

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, lossFunctionDerivativeVector)

end

function LinearRegressionCovariancePreconditionedVariantModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function LinearRegressionCovariancePreconditionedVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewLinearRegressionCovariancePreconditionedVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewLinearRegressionCovariancePreconditionedVariantModel, LinearRegressionCovariancePreconditionedVariantModel)
	
	NewLinearRegressionCovariancePreconditionedVariantModel:setName("LinearRegressionCovariancePreconditionedVariant")

	NewLinearRegressionCovariancePreconditionedVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewLinearRegressionCovariancePreconditionedVariantModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewLinearRegressionCovariancePreconditionedVariantModel.Regularizer = parameterDictionary.Regularizer

	return NewLinearRegressionCovariancePreconditionedVariantModel

end

function LinearRegressionCovariancePreconditionedVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function LinearRegressionCovariancePreconditionedVariantModel:train(featureMatrix, labelVector)

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

	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
	self.pMatrix = calculatePMatrix(featureMatrix)

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
	
	self.pMatrix = nil

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	return costArray

end

function LinearRegressionCovariancePreconditionedVariantModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return LinearRegressionCovariancePreconditionedVariantModel
