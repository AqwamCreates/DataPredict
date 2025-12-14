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

local NegativeBinomialRegressionModel = {}

NegativeBinomialRegressionModel.__index = NegativeBinomialRegressionModel

setmetatable(NegativeBinomialRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.5

local defaultDispersion = 0.5 -- alpha

local defaultEpsilon = 1e-14

-- Approximate Gamma function (Stirling's approximation).

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

function NegativeBinomialRegressionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local dispersion = self.dispersion  -- Overdispersion parameter (α > 0)
	local epsilon = self.epsilon or 1e-8

	hypothesisVector = AqwamTensorLibrary:add(hypothesisVector, epsilon)
	
	labelVector = AqwamTensorLibrary:add(labelVector, epsilon)

	local numberOfData = #labelVector
	
	local muVector = hypothesisVector
	
	local theta = 1 / dispersion

	-- Part 1: log Γ(y_i + θ) terms.
	
	local yPlusThetaVector = AqwamTensorLibrary:add(labelVector, theta)
	
	local logGammaYPlusTheta = AqwamTensorLibrary:applyFunction(function(x) return math.log(approximateGammaFunction(x)) end, yPlusThetaVector)

	-- Part 2: -log Γ(θ) terms (constant for all i, computed once).
	
	local logGammaTheta = math.log(approximateGammaFunction(theta))

	-- Part 3: θ * log(θ) terms (constant).
	
	local thetalogTheta = theta * math.log(theta)

	-- Part 4: (θ + y_i) * log(θ + μ_i).
	
	local thetaPlusMu = AqwamTensorLibrary:add(muVector, theta)
	
	local logThetaPlusMu = AqwamTensorLibrary:applyFunction(math.log, thetaPlusMu)
	
	local thetaPlusY = AqwamTensorLibrary:add(labelVector, theta)
	
	local costVectorPart4 = AqwamTensorLibrary:multiply(thetaPlusY, logThetaPlusMu)

	-- Part 5: -y_i * log(μ_i).
	
	local logMu = AqwamTensorLibrary:applyFunction(math.log, muVector)
	
	local costVectorPart5 = AqwamTensorLibrary:multiply(labelVector, logMu)
	
	costVectorPart5 = AqwamTensorLibrary:multiplyScalar(costVectorPart5, -1)

	local costVector = AqwamTensorLibrary:add(costVectorPart4, costVectorPart5)

	-- Add constants: -log Γ(y_i + θ) + log Γ(θ) - θ * log(θ).
	
	costVector = AqwamTensorLibrary:add(costVector, logGammaTheta - thetalogTheta)
	
	costVector = AqwamTensorLibrary:subtract(costVector, logGammaYPlusTheta)

	-- Sum over all data points.
	
	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	--[[

		Add log factorial term log Γ(y_i + 1) if needed (but often omitted as constant)
		
		For completeness (optional):
		
		local yPlusOneVector = AqwamTensorLibrary:add(labelVector, 1)
		
		local logGammaYPlusOneVector = AqwamTensorLibrary:applyFunction(function(x) return math.log(approximateGammaFunction(x)) end, yPlusOneVector)
		
		local logFactorialSum = AqwamTensorLibrary:sum(logGammaYPlusOneVector)
		
		totalCost = totalCost + logFactorialSum
	
	--]]

	local Regularizer = self.Regularizer
	
	if (Regularizer) then
		
		totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters)
		
	end

	local averageCost = -totalCost / numberOfData

	return averageCost
end

function NegativeBinomialRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)
	
	local exponentTermVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	local hypothesisVector = AqwamTensorLibrary:applyFunction(math.exp, exponentTermVector)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function NegativeBinomialRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	local lossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function NegativeBinomialRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

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

function NegativeBinomialRegressionModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function NegativeBinomialRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewNegativeBinomialRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewNegativeBinomialRegressionModel, NegativeBinomialRegressionModel)
	
	NewNegativeBinomialRegressionModel:setName("NegativeBinomialRegression")

	NewNegativeBinomialRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewNegativeBinomialRegressionModel.dispersion = parameterDictionary.dispersion or defaultDispersion
	
	NewNegativeBinomialRegressionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewNegativeBinomialRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewNegativeBinomialRegressionModel.Regularizer = parameterDictionary.Regularizer

	return NewNegativeBinomialRegressionModel

end

function NegativeBinomialRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function NegativeBinomialRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function NegativeBinomialRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local dispersion = self.dispersion

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

		local lossGradientVectorPart1 = AqwamTensorLibrary:subtract(hypothesisVector, labelVector)
		
		local lossGradientVectorPart2 = AqwamTensorLibrary:multiply(dispersion, hypothesisVector)
		
		local lossGradientVectorPart3 = AqwamTensorLibrary:add(1, lossGradientVectorPart2)
		
		local lossGradientVector = AqwamTensorLibrary:divide(lossGradientVectorPart1, lossGradientVectorPart3)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function NegativeBinomialRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local exponentTermVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	local predictedVector = AqwamTensorLibrary:applyFunction(math.exp, exponentTermVector)

	return predictedVector

end

return NegativeBinomialRegressionModel
