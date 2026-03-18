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

local Solvers = script.Parent.Parent.Solvers

local SupportVectorMachineGradientVariantModel = {}

SupportVectorMachineGradientVariantModel.__index = SupportVectorMachineGradientVariantModel

setmetatable(SupportVectorMachineGradientVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCValue = 1

local defaultCostFunction = "SquaredHingeLoss"

local defaultSolver = "GaussNewton"

local lossFunctionList = {

	["HingeLoss"] = function (h, y) return math.max(0, (1 - (h * y))) end,

	["SquaredHingeLoss"] = function (h, y) return math.pow(math.max(0, (1 - (h * y))), 2) end,

}

local lossFunctionGradientList = {

	["HingeLoss"] = function (h, y)

		local scale = (((h * y) < 1) and 1) or 0

		return -(y * scale)

	end,

	["SquaredHingeLoss"] = function (h, y)

		local margin = 1 - (h * y)

		local scale = ((margin > 0) and margin) or 0

		return -(2 * y * scale)

	end,

}

local function hingeFunction(value)
	
	return math.max(0, value)
	
end

local function misclassificationMaskFunction(value)
	
	return (value < 1) and 1 or 0
	
end

local function seperatorFunction(x) 

	return ((x > 0) and 1) or ((x < 0) and -1) or 0

end

function SupportVectorMachineGradientVariantModel:calculateCost(hypothesisVector, labelVector, hasBias)

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)

	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters, hasBias) end

	local averageCost = (self.cValue * totalCost) / #labelVector

	return averageCost

end

function SupportVectorMachineGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorMachineGradientVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local lossFunctionDerivativeVector = self.Solver:calculate(self.ModelParameters, self.featureMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function SupportVectorMachineGradientVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end
	
	local ModelParameters = self.ModelParameters
	
	local Regularizer = self.Regularizer
	
	local Optimizer = self.Optimizer
	
	local learningRate = self.learningRate

	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters, hasBias)

		lossFunctionDerivativeVector = AqwamTensorLibrary:add(lossFunctionDerivativeVector, regularizationDerivatives)

	end

	lossFunctionDerivativeVector = AqwamTensorLibrary:divide(lossFunctionDerivativeVector, numberOfData)

	if (Optimizer) then 

		lossFunctionDerivativeVector = Optimizer:calculate(learningRate, lossFunctionDerivativeVector, ModelParameters) 

	else

		lossFunctionDerivativeVector = AqwamTensorLibrary:multiply(learningRate, lossFunctionDerivativeVector)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, lossFunctionDerivativeVector)

end

function SupportVectorMachineGradientVariantModel:update(lossGradientVector, hasBias, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function SupportVectorMachineGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorMachineGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorMachineGradientVariantModel, SupportVectorMachineGradientVariantModel)
	
	NewSupportVectorMachineGradientVariantModel:setName("SupportVectorMachineGradientVariant")

	NewSupportVectorMachineGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorMachineGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue
	
	NewSupportVectorMachineGradientVariantModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewSupportVectorMachineGradientVariantModel.Optimizer = parameterDictionary.Optimizer

	NewSupportVectorMachineGradientVariantModel.Regularizer = parameterDictionary.Regularizer
	
	NewSupportVectorMachineGradientVariantModel.Solver = parameterDictionary.Solver or require(Solvers[defaultSolver]).new()

	return NewSupportVectorMachineGradientVariantModel

end

function SupportVectorMachineGradientVariantModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function SupportVectorMachineGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorMachineGradientVariantModel:setSolver(Solver)

	self.Solver = Solver

end

function SupportVectorMachineGradientVariantModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue

	local Optimizer = self.Optimizer

	local hasBias = self:checkIfFeatureMatrixHasBias(featureMatrix)

	local costArray = {}

	local numberOfIterations = 0
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local hypothesisVector = self:calculateHypothesisVector(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisVector, labelVector, hasBias)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossGradientVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisVector, labelVector)
		
		lossGradientVector = AqwamTensorLibrary:multiply(cValue, lossGradientVector)

		self:update(lossGradientVector, hasBias, true)

	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) or self:checkIfNan(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	if (self.autoResetConvergenceCheck) then self:resetConvergenceCheck() end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end
	
	if (self.autoResetSolvers) then self.Solver:reset() end

	return costArray

end

function SupportVectorMachineGradientVariantModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	if (returnOriginalOutput) then return predictedVector end

	return AqwamTensorLibrary:applyFunction(seperatorFunction, predictedVector)

end

return SupportVectorMachineGradientVariantModel
