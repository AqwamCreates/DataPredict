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

local Solvers = script.Parent.Parent.Solvers

local RankingSupportVectorMachineGradientVariantModel = {}

RankingSupportVectorMachineGradientVariantModel.__index = RankingSupportVectorMachineGradientVariantModel

setmetatable(RankingSupportVectorMachineGradientVariantModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCValue = 1

local defaultSolver = "GaussNewton"

local function hingeFunction(value)
	
	return math.max(0, value)
	
end

local function misclassificationMaskFunction(value)
	
	return (value < 1) and 1 or 0
	
end

local function seperatorFunction(x) 

	return ((x > 0) and 1) or ((x < 0) and -1) or 0

end

local function convertDatasetToPairedComparisonDataset(featureMatrix, labelVector)

	local pairedComparisonFeatureMatrix = {}
	
	local currentComparisonCount = 0
	
	local primaryFeatureVector
	
	local primaryLabelValue
	
	local secondaryFeatureVector
	
	for i, unwrappedPrimaryFeatureVector in ipairs(featureMatrix) do
	
		primaryFeatureVector = {unwrappedPrimaryFeatureVector}
		
		primaryLabelValue = labelVector[i][1]
	
		for j, unwrappedSecondaryFeatureVector in ipairs(featureMatrix) do
		
			if (i ~= j) then
			
				if (primaryLabelValue > labelVector[j][1]) then
				
					currentComparisonCount = currentComparisonCount + 1
					
					secondaryFeatureVector = {unwrappedSecondaryFeatureVector}
					
					pairedComparisonFeatureMatrix[currentComparisonCount] = AqwamTensorLibrary:subtract(primaryFeatureVector, secondaryFeatureVector)[1]
				
				end
			
			end
		
		end
	
	end
	
	local pairedComparisonLabelVector = AqwamTensorLibrary:createTensor({#pairedComparisonFeatureMatrix, 1}, 1)

	return pairedComparisonFeatureMatrix, pairedComparisonLabelVector

end

function RankingSupportVectorMachineGradientVariantModel:calculateCost(hypothesisVector, labelVector, hasBias)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local oneVector = AqwamTensorLibrary:createTensor({#labelVector, 1}, 1)
	
	local marginVector = AqwamTensorLibrary:multiply(labelVector, hypothesisVector)
	
	local hingeVector = AqwamTensorLibrary:subtract(oneVector, marginVector)
	
	local costVector = AqwamTensorLibrary:applyFunction(hingeFunction, hingeVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters, hasBias) end

	local averageCost = (self.cValue * totalCost) / #labelVector

	return averageCost

end

function RankingSupportVectorMachineGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function RankingSupportVectorMachineGradientVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local lossFunctionDerivativeVector = self.Solver:calculate(self.ModelParameters, self.featureMatrix, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function RankingSupportVectorMachineGradientVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

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

function RankingSupportVectorMachineGradientVariantModel:update(lossGradientVector, hasBias, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (clearAllMatrices) then 

		self.featureMatrix = nil 

		self.lossFunctionDerivativeVector = nil

	end

end

function RankingSupportVectorMachineGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorMachineGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorMachineGradientVariantModel, RankingSupportVectorMachineGradientVariantModel)
	
	NewSupportVectorMachineGradientVariantModel:setName("SupportVectorMachineGradientVariant")

	NewSupportVectorMachineGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorMachineGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue

	NewSupportVectorMachineGradientVariantModel.Optimizer = parameterDictionary.Optimizer

	NewSupportVectorMachineGradientVariantModel.Regularizer = parameterDictionary.Regularizer
	
	NewSupportVectorMachineGradientVariantModel.Solver = parameterDictionary.Solver or require(Solvers[defaultSolver]).new()

	return NewSupportVectorMachineGradientVariantModel

end

function RankingSupportVectorMachineGradientVariantModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function RankingSupportVectorMachineGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function RankingSupportVectorMachineGradientVariantModel:setSolver(Solver)

	self.Solver = Solver

end

function RankingSupportVectorMachineGradientVariantModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local cValue = self.cValue

	local Optimizer = self.Optimizer

	local hasBias = self:checkIfFeatureMatrixHasBias(featureMatrix)

	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
	featureMatrix, labelVector = convertDatasetToPairedComparisonDataset(featureMatrix, labelVector)

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

		local marginVector = AqwamTensorLibrary:multiply(labelVector, hypothesisVector)
		
		local misclassifiedMaskVector = AqwamTensorLibrary:applyFunction(misclassificationMaskFunction, marginVector)
		
		local lossGradientVector = AqwamTensorLibrary:multiply(-cValue, labelVector, misclassifiedMaskVector)

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

function RankingSupportVectorMachineGradientVariantModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	if (returnOriginalOutput) then return predictedVector end

	return AqwamTensorLibrary:applyFunction(seperatorFunction, predictedVector)

end

return RankingSupportVectorMachineGradientVariantModel		local sumSquaredSlackVariableValue = squaredPositiveSlackVariableValue + squaredNegativeSlackVariableValue

		return sumSquaredSlackVariableValue 

	end,

}

local lossFunctionGradientList = {

	["EpsilonInsensitiveLoss"] = function (h, y, epsilon)
		
		local errorValue = h - y
		
		if (errorValue > epsilon) then

			return (errorValue - epsilon)

		elseif (errorValue < -epsilon) then

			return (errorValue + epsilon)

		else

			return 0

		end
		
	end,

	["SquaredEpsilonInsensitiveLoss"] = function (h, y, epsilon)
		
		local errorValue = h - y

		if (errorValue > epsilon) then
			
			return 2 * (errorValue - epsilon)
			
		elseif (errorValue < -epsilon) then
			
			return 2 * (errorValue + epsilon)
			
		else
			
			return 0
			
		end

	end,

}

function SupportVectorRegressionGradientVariantModel:calculateCost(hypothesisVector, labelVector, hasBias)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local epsilon = self.epsilon
	
	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector, {{epsilon}})

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	totalCost = self.cValue * totalCost
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters, hasBias) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function SupportVectorRegressionGradientVariantModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then self.featureMatrix = featureMatrix end

	return hypothesisVector

end

function SupportVectorRegressionGradientVariantModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local lossFunctionDerivativeVector = self.Solver:calculate(self.ModelParameters, self.featureMatrix, nil, lossGradientVector)

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVector = lossFunctionDerivativeVector end

	return lossFunctionDerivativeVector

end

function SupportVectorRegressionGradientVariantModel:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

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

function SupportVectorRegressionGradientVariantModel:update(lossGradientVector, hasBias, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

	if (clearAllMatrices) then 

		self.featureMatrix = nil

		self.lossFunctionDerivativeVector = nil

	end

end

function SupportVectorRegressionGradientVariantModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSupportVectorRegressionGradientVariantModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewSupportVectorRegressionGradientVariantModel, SupportVectorRegressionGradientVariantModel)
	
	NewSupportVectorRegressionGradientVariantModel:setName("SupportVectorRegressionGradientVariant")

	NewSupportVectorRegressionGradientVariantModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewSupportVectorRegressionGradientVariantModel.cValue = parameterDictionary.cValue or defaultCValue
	
	NewSupportVectorRegressionGradientVariantModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewSupportVectorRegressionGradientVariantModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewSupportVectorRegressionGradientVariantModel.Optimizer = parameterDictionary.Optimizer

	NewSupportVectorRegressionGradientVariantModel.Regularizer = parameterDictionary.Regularizer
	
	NewSupportVectorRegressionGradientVariantModel.Solver = parameterDictionary.Solver or require(Solvers[defaultSolver]).new({isLinear = true})

	return NewSupportVectorRegressionGradientVariantModel

end

function SupportVectorRegressionGradientVariantModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function SupportVectorRegressionGradientVariantModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function SupportVectorRegressionGradientVariantModel:setSolver(Solver)

	self.Solver = Solver

end

function SupportVectorRegressionGradientVariantModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
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
	
	local epsilon = self.epsilon

	local Optimizer = self.Optimizer
	
	local hasBias = self:checkIfFeatureMatrixHasBias(featureMatrix)
	
	local epsilonVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, epsilon)
	
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

		local lossGradientVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisVector, labelVector, epsilonVector)
		
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

function SupportVectorRegressionGradientVariantModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})
		
		self.ModelParameters = ModelParameters
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	return predictedVector

end

return SupportVectorRegressionGradientVariantModel
