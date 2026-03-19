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

local TweedieRegressionModel = {}

TweedieRegressionModel.__index = TweedieRegressionModel

setmetatable(TweedieRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultPower = 1.5

local defaultSolver = "GaussNewton"

local function tweedieLossFunctionToApply(h, y, power)
	
	if (power == 0) then return math.pow((h - y), 2) end -- Linear special case.
	
	if (power == 1) then return (2 * (y * math.log(y / h) - y + h)) end -- Poisson special case.
	
	if (power == 2) then -- Gamma special case.
		
		local ratio = y / h

		return (2 * ((y * math.log(ratio)) + ratio - 1))
		
	end
	
	local oneMinusPower = 1 - power
	
	local twoMinusPower = 2 - power
	
	local tweedieLossValuePart1 = math.pow(y, twoMinusPower) / (oneMinusPower * twoMinusPower)
	
	local tweedieLossValuePart2 = (y * math.pow(h, oneMinusPower)) / oneMinusPower
		
	local tweedieLossValuePart3 = math.pow(h, twoMinusPower) / twoMinusPower
	
	local tweedieLossValue = 2 * (tweedieLossValuePart1 - tweedieLossValuePart2 + tweedieLossValuePart3)
	
	return tweedieLossValue
	
end

local function tweedieLossFunctionGradientToApply(h, y, power)
	
	if (power == 0) then return (2 * (h - y)) end -- Linear special case.
	
	if (power == 1) then return (2 * (1 - (y / h))) end -- Poisson special case.
	
	if (power == 2) then return (2 * ((h - y) / math.pow(h, 2))) end -- Gamma special case.
	
	local tweedieLossGradientValuePart1 = math.pow(h, (1 - power))
	
	local tweedieLossGradientValuePart2 = y * math.pow(h, -power)
	
	local tweedieLossGradientValue = 2 * (tweedieLossGradientValuePart1 - tweedieLossGradientValuePart2)
	
	return tweedieLossGradientValue
	
end

function TweedieRegressionModel:calculateCost(hypothesisVector, labelVector, hasBias)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local power = self.power

	local costVector = AqwamTensorLibrary:applyFunction(tweedieLossFunctionToApply, hypothesisVector, labelVector, {{power}})

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters, hasBias) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function TweedieRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local zVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	local hypothesisVector = AqwamTensorLibrary:applyFunction(math.exp, zVector)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function TweedieRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local lossFunctionDerivativeVector = self.Solver:calculate(self.ModelParameters, self.featureMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function TweedieRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

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

function TweedieRegressionModel:update(lossGradientVector, hasBias, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (clearAllMatrices) then 

		self.featureMatrix = nil

		self.lossFunctionDerivativeVector = nil

	end

end

function TweedieRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewTweedieRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewTweedieRegressionModel, TweedieRegressionModel)
	
	NewTweedieRegressionModel:setName("TweedieRegression")

	NewTweedieRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewTweedieRegressionModel.power = parameterDictionary.power or defaultPower

	NewTweedieRegressionModel.Optimizer = parameterDictionary.Optimizer

	NewTweedieRegressionModel.Regularizer = parameterDictionary.Regularizer
	
	NewTweedieRegressionModel.Solver = parameterDictionary.Solver or require(Solvers[defaultSolver]).new()

	return NewTweedieRegressionModel

end

function TweedieRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function TweedieRegressionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function TweedieRegressionModel:setSolver(Solver)

	self.Solver = Solver

end

function TweedieRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (numberOfFeatures ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local power = self.power

	local Optimizer = self.Optimizer
	
	local hasBias = self:checkIfFeatureMatrixHasBias(featureMatrix)
	
	local powerVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, power)

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

		local lossGradientVector = AqwamTensorLibrary:applyFunction(tweedieLossFunctionGradientToApply, hypothesisVector, labelVector, powerVector)
		
		lossGradientVector = AqwamTensorLibrary:multiply(lossGradientVector, hypothesisVector) -- Since the derivative of exponential is itself, we can reuse the hypothesis vector.

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

function TweedieRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	return self:calculateHypothesisVector(featureMatrix, false)

end

return TweedieRegressionModel
