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

local SupportVectorMachineNaturalGradientVariantModel = {}

SupportVectorMachineNaturalGradientVariantModel.__index = SupportVectorMachineNaturalGradientVariantModel

setmetatable(SupportVectorMachineNaturalGradientVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 1

local defaultCValue = 1

local function hingeFunction(value)
	
	return math.max(0, value)
	
end

local function misclassificationMaskFunction(value)
	
	return (value < 1) and 1 or 0
	
end

local function seperatorFunction(x) 

	return ((x > 0) and 1) or ((x < 0) and -1) or 0

end

local function calculatePMatrix(featureMatrix)

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local pMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)

	pMatrix = AqwamTensorLibrary:inverse(pMatrix)

	pMatrix = AqwamTensorLibrary:dotProduct(pMatrix, transposedFeatureMatrix)

	return pMatrix

end

function SupportVectorMachineNaturalGradientVariantModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local oneVector = AqwamTensorLibrary:createTensor({#labelVector, 1}, 1)
	
	local marginVector = AqwamTensorLibrary:multiply(labelVector, hypothesisVector)
	
	local hingeVector = AqwamTensorLibrary:subtract(oneVector, marginVector)
	
	local costVector = AqwamTensorLibrary:applyFunction(hingeFunction, hingeVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = (self.cValue * totalCost) / #labelVector

	return averageCost

end

function SupportVectorMachineNaturalGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorMachineNaturalGradientVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix
	
	local pMatrix = self.pMatrix or calculatePMatrix(featureMatrix)

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(pMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function SupportVectorMachineNaturalGradientVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

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

function SupportVectorMachineNaturalGradientVariantModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function SupportVectorMachineNaturalGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorMachineNaturalGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorMachineNaturalGradientVariantModel, SupportVectorMachineNaturalGradientVariantModel)
	
	NewSupportVectorMachineNaturalGradientVariantModel:setName("SupportVectorMachineNaturalGradientVariant")

	NewSupportVectorMachineNaturalGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorMachineNaturalGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue

	NewSupportVectorMachineNaturalGradientVariantModel.Regularizer = parameterDictionary.Regularizer

	return NewSupportVectorMachineNaturalGradientVariantModel

end

function SupportVectorMachineNaturalGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorMachineNaturalGradientVariantModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue

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

		local marginVector = AqwamTensorLibrary:multiply(labelVector, hypothesisVector)
		
		local misclassifiedMaskVector = AqwamTensorLibrary:applyFunction(misclassificationMaskFunction, marginVector)
		
		local lossGradientVector = AqwamTensorLibrary:multiply(-cValue, labelVector, misclassifiedMaskVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	self.pMatrix = nil

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	return costArray

end

function SupportVectorMachineNaturalGradientVariantModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	if (returnOriginalOutput) then return predictedVector end

	return AqwamTensorLibrary:applyFunction(seperatorFunction, predictedVector)

end

return SupportVectorMachineNaturalGradientVariantModel
