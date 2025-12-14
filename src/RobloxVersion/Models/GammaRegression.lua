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

local GammaRegressionModel = {}

GammaRegressionModel.__index = GammaRegressionModel

setmetatable(GammaRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultShape = 1 -- alpha

local defaultEpsilon = 1e-14

-- Approximate gamma function (Stirling's approximation).

local function approximateGammaFunction(x)
	
	if (x <= 0) then return math.huge end
	
	if (x == 1) then return 1 end
	
	if (x == 0.5) then return math.sqrt(math.pi) end

	-- Stirling's approximation for large x.
	
	if (x > 10) then return math.sqrt(2 * math.pi / x) * ((x / math.exp(1)) ^ x) end

	-- Recurrence for smaller x.
	
	local result = 1
	
	while (x > 2) do
		
		x = x - 1
		
		result = result * x
		
	end
	
	while (x < 1) do
		
		result = result / x
		
		x = x + 1
		
	end
	
	return result
	
end

function GammaRegressionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local shape = self.shape
	
	local epsilon = self.epsilon
	
	hypothesisVector = AqwamTensorLibrary:add(hypothesisVector, epsilon)
	
	labelVector = AqwamTensorLibrary:add(labelVector, epsilon)
	
	local thetaVector = AqwamTensorLibrary:divide(hypothesisVector, shape)
	
	local logShape = math.log(shape)
	
	local logHypothesisVector = AqwamTensorLibrary:applyFunction(math.log, hypothesisVector)
	
	local costVectorPart1 = AqwamTensorLibrary:subtract(logShape, logHypothesisVector)
	
	costVectorPart1 = AqwamTensorLibrary:multiply(costVectorPart1, shape)
	
	local logLabelVector = AqwamTensorLibrary:applyFunction(math.log, labelVector)
	
	local costVectorPart2 = AqwamTensorLibrary:multiply(logLabelVector, (1 - shape))
	
	local labelVectorDividedByHypothesisVector = AqwamTensorLibrary:divide(labelVector, hypothesisVector)
	
	local costVectorPart3 = AqwamTensorLibrary:multiply(labelVectorDividedByHypothesisVector, shape)
	
	local costVector = AqwamTensorLibrary:add(costVectorPart1, costVectorPart2)
	
	costVector = AqwamTensorLibrary:add(costVector, costVectorPart3)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local gamma = approximateGammaFunction(shape)
	
	local logGamma = math.log(gamma)
	
	local numberOfData = #labelVector
	
	totalCost = totalCost + (logGamma * numberOfData)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / numberOfData

	return averageCost

end

function GammaRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)
	
	local exponentTermVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	local hypothesisVector = AqwamTensorLibrary:applyFunction(math.exp, exponentTermVector)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function GammaRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function GammaRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

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

function GammaRegressionModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function GammaRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewGammaRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewGammaRegressionModel, GammaRegressionModel)
	
	NewGammaRegressionModel:setName("GammaRegression")

	NewGammaRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewGammaRegressionModel.shape = parameterDictionary.shape or defaultShape
	
	NewGammaRegressionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewGammaRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewGammaRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewGammaRegressionModel

end

function GammaRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function GammaRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function GammaRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local shape = self.shape

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
		
		lossGradientVector = AqwamTensorLibrary:divide(lossGradientVector, hypothesisVector)
		
		lossGradientVector = AqwamTensorLibrary:multiply(lossGradientVector, shape)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function GammaRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local exponentTermVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	local predictedVector = AqwamTensorLibrary:applyFunction(math.exp, exponentTermVector)

	return predictedVector

end

return GammaRegressionModel
