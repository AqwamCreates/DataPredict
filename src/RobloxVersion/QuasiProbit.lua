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

local QuasiProbitRegressionModel = {}

QuasiProbitRegressionModel.__index = QuasiProbitRegressionModel

setmetatable(QuasiProbitRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "Sigmoid"

local function cutOffFunction(value)
	
	return (value >= 0.5) and 1 or 0
	
end

local function calculateLogLikelihood(hypothesisValue, labelValue)
	
	return (labelValue * math.log(hypothesisValue)) + ((1 - labelValue) * math.log(1 - hypothesisValue))
	
end

function QuasiProbitRegressionModel:calculateCost(hypothesisVector, labelVector)

	local costVector = AqwamTensorLibrary:applyFunction(calculateLogLikelihood, hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = -totalCost / #labelVector

	return averageCost

end

function QuasiProbitRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

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

	return hypothesisVector

end

function QuasiProbitRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientVector)

	if (self.areGradientsSaved) then self.Gradients = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function QuasiProbitRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

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

function QuasiProbitRegressionModel:update(lossGradientVector, clearFeatureMatrix)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)
	
	if (clearFeatureMatrix) then self.featureMatrix = nil end

end

function QuasiProbitRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewQuasiProbitRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewQuasiProbitRegressionModel, QuasiProbitRegressionModel)
	
	NewQuasiProbitRegressionModel:setName("ProbitRegression")

	NewQuasiProbitRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewQuasiProbitRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewQuasiProbitRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewQuasiProbitRegressionModel

end

function QuasiProbitRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function QuasiProbitRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function QuasiProbitRegressionModel:train(featureMatrix, labelVector)

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

function QuasiProbitRegressionModel:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1}) end

	local outputVector = self:calculateHypothesisVector(featureMatrix, false)

	if (returnOriginalOutput) then return outputVector end

	local predictedLabelVector = AqwamTensorLibrary:applyFunction(cutOffFunction, outputVector)

	return predictedLabelVector, outputVector

end

return QuasiProbitRegressionModel
