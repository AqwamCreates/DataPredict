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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

local SupportVectorRegressionNaturalGradientVariantModel = {}

SupportVectorRegressionNaturalGradientVariantModel.__index = SupportVectorRegressionNaturalGradientVariantModel

setmetatable(SupportVectorRegressionNaturalGradientVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 1

local defaultCValue = 1

local defaultEpsilon = 1

local function calculatePMatrix(featureMatrix)

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local pMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	pMatrix = AqwamTensorLibrary:inverse(pMatrix)

	pMatrix = AqwamTensorLibrary:dotProduct(pMatrix, transposedFeatureMatrix)

	return pMatrix

end

function SupportVectorRegressionNaturalGradientVariantModel:calculateCost(hypothesisVector, labelVector)

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

function SupportVectorRegressionNaturalGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorRegressionNaturalGradientVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix
	
	local pMatrix = self.pMatrix or calculatePMatrix(featureMatrix)

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(pMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function SupportVectorRegressionNaturalGradientVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

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

function SupportVectorRegressionNaturalGradientVariantModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function SupportVectorRegressionNaturalGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorRegressionNaturalGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorRegressionNaturalGradientVariantModel, SupportVectorRegressionNaturalGradientVariantModel)
	
	NewSupportVectorRegressionNaturalGradientVariantModel:setName("SupportVectorRegressionNaturalGradientVariant")

	NewSupportVectorRegressionNaturalGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorRegressionNaturalGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue
	
	NewSupportVectorRegressionNaturalGradientVariantModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewSupportVectorRegressionNaturalGradientVariantModel.Regularizer = parameterDictionary.Regularizer

	return NewSupportVectorRegressionNaturalGradientVariantModel

end

function SupportVectorRegressionNaturalGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorRegressionNaturalGradientVariantModel:train(featureMatrix, labelVector)

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
	
	local functionToApply = function(errorValue) return ((errorValue > epsilon) and (errorValue - epsilon)) or ((errorValue < -epsilon) and (errorValue + epsilon)) or 0 end

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
		
		local errorVector = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)

		local lossGradientVector = AqwamTensorLibrary:applyFunction(functionToApply, errorVector)
		
		lossGradientVector = AqwamTensorLibrary:multiply(cValue, lossGradientVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	self.pMatrix = nil

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	return costArray

end

function SupportVectorRegressionNaturalGradientVariantModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return SupportVectorRegressionNaturalGradientVariantModel
