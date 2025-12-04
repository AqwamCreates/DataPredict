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

local ZTableFunction = require(script.Parent.Parent.Cores.ZTableFunction)

local ProbitRegressionModel = {}

ProbitRegressionModel.__index = ProbitRegressionModel

setmetatable(ProbitRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local function gradientFunction(hypothesisValue, labelValue, zValue)
	
	local probabilityDensityFunctionValue = math.exp(-0.5 * math.pow(zValue, 2)) / math.sqrt(2 * math.pi)

	return (probabilityDensityFunctionValue * ((labelValue / hypothesisValue) - ((1 - labelValue) / (1 - hypothesisValue))))
	
end

local function calculateLogLikelihood(hypothesisValue, labelValue)
	
	return (labelValue * math.log(hypothesisValue)) + ((1 - labelValue) * math.log(1 - hypothesisValue))
	
end

local function cutOffFunction(value)

	return (value >= 0.5) and 1 or 0

end
	
function ProbitRegressionModel:calculateCost(hypothesisVector, labelVector)

	local costVector = AqwamTensorLibrary:applyFunction(calculateLogLikelihood, hypothesisVector, labelVector)

	local totalCost = -AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function ProbitRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local zVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then 

		self.featureMatrix = featureMatrix

	end

	local hypothesisVector = {}
	
	local zValue
	
	local cumulativeDistributionValue
	
	for i, unwrappedZVector in ipairs(zVector) do
		
		zValue = math.clamp(unwrappedZVector[1], -3.9, 3.9)
		
		cumulativeDistributionValue = ZTableFunction:getStandardNormalCumulativeDistributionFunction(zValue)
		
		hypothesisVector[i] = {cumulativeDistributionValue}
		
	end

	return hypothesisVector, zVector

end

function ProbitRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientVector)

	if (self.areGradientsSaved) then self.Gradients = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function ProbitRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end
	
	local ModelParameters = self.ModelParameters

	local Regularizer = self.Regularizer

	local Optimizer = self.Optimizer

	local learningRate = self.learningRate
	
	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

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

function ProbitRegressionModel:update(lossGradientVector, clearFeatureMatrix)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)
	
	if (clearFeatureMatrix) then self.featureMatrix = nil end

end

function ProbitRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewProbitRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewProbitRegressionModel, ProbitRegressionModel)
	
	NewProbitRegressionModel:setName("ProbitRegression")

	NewProbitRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewProbitRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewProbitRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewProbitRegressionModel

end

function ProbitRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function ProbitRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function ProbitRegressionModel:train(featureMatrix, labelVector)

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

		local hypothesisVector, zVector = self:calculateHypothesisVector(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisVector, labelVector)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossGradientVector = AqwamTensorLibrary:applyFunction(gradientFunction, hypothesisVector, labelVector, zVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then
		
		if (cost == math.huge) then warn("The model diverged.") end
		
		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end
		
	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function ProbitRegressionModel:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1}) end

	local outputVector = self:calculateHypothesisVector(featureMatrix, false)

	if (returnOriginalOutput) then return outputVector end

	local predictedLabelVector = AqwamTensorLibrary:applyFunction(cutOffFunction, outputVector)

	return predictedLabelVector, outputVector

end

return ProbitRegressionModel
