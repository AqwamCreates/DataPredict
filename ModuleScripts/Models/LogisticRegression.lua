local BaseModel = require(script.Parent.BaseModel)

LogisticRegressionModel = {}

LogisticRegressionModel.__index = LogisticRegressionModel

setmetatable(LogisticRegressionModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "sigmoid"

local defaultTargetCost = 0

local sigmoidFunctionList = {

	["sigmoid"] = function (z) return 1/(1+math.exp(-1 * z)) end,

}

local lossFunctionList = {
	
	["sigmoid"] = function (y, h) return -(y * math.log10(h) + (1 - y) * math.log10(1 - h)) end
	
	
}

local function calculateHypothesisVector(featureMatrix, modelParameters, sigmoidFunction)
	
	local numberOfData = #featureMatrix
	
	local zVector = AqwamMatrixLibrary:dotProduct(featureMatrix, modelParameters)
	
	if (type(zVector) == "number") then zVector = {{zVector}} end
		
	local result = AqwamMatrixLibrary:applyFunction(sigmoidFunctionList[sigmoidFunction], zVector)
	
	return result
	
end

local function calculateCost(modelParameters, featureMatrix, labelVector, sigmoidFunction)
	
	local numberOfData = #featureMatrix
	
	local hypothesisVector = calculateHypothesisVector(featureMatrix, modelParameters, sigmoidFunction)
	
	local costVector = AqwamMatrixLibrary:applyFunction(lossFunctionList[sigmoidFunction], labelVector, hypothesisVector)

	local totalCost = AqwamMatrixLibrary:sum(costVector)
	
	local averageCost = totalCost / numberOfData
	
	return averageCost
	
end

local function gradientDescent(modelParameters, featureMatrix, labelVector, sigmoidFunction)
	
	local numberOfData = #featureMatrix

	local hypothesisVector = calculateHypothesisVector(featureMatrix, modelParameters, sigmoidFunction)

	local calculatedError = AqwamMatrixLibrary:subtract(hypothesisVector, labelVector)
	
	local calculatedErrorWithFeatureMatrix = AqwamMatrixLibrary:dotProduct(AqwamMatrixLibrary:transpose(featureMatrix), calculatedError)

	local costFunctionDerivatives = AqwamMatrixLibrary:multiply((1/numberOfData),  calculatedErrorWithFeatureMatrix)
	
	return costFunctionDerivatives
	
end

function LogisticRegressionModel.new(maxNumberOfIterations, learningRate, sigmoidFunction, targetCost)
	
	local NewLogisticRegressionModel = BaseModel.new()

	setmetatable(NewLogisticRegressionModel, LogisticRegressionModel)

	NewLogisticRegressionModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewLogisticRegressionModel.learningRate = learningRate or defaultLearningRate

	NewLogisticRegressionModel.sigmoidFunction = sigmoidFunction or defaultSigmoidFunction

	NewLogisticRegressionModel.targetCost = targetCost or defaultTargetCost

	NewLogisticRegressionModel.validationFeatureMatrix = nil

	NewLogisticRegressionModel.validationLabelVector = nil
	
	NewLogisticRegressionModel.Optimizer = nil
	
	NewLogisticRegressionModel.Regularization = nil
	
	return NewLogisticRegressionModel
	
end

function LogisticRegressionModel:setParameters(maxNumberOfIterations, learningRate, sigmoidFunction, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.sigmoidFunction = sigmoidFunction or self.sigmoidFunction

	self.targetCost = targetCost or self.targetCost

end

function LogisticRegressionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function LogisticRegressionModel:setRegularization(Regularization)

	self.Regularization = Regularization

end

function LogisticRegressionModel:train(featureMatrix, labelVector)

	local cost
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local costFunctionDerivatives
	
	local numberOfData = #featureMatrix[1]
	
	local previousCostFunctionDerivatives
	
	local lambda
	
	local regularizationCost
	
	local RegularizationDerivatives
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end

	if (self.ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], 1)

	end
	
	repeat
		
		self:iterationWait()
		
		numberOfIterations += 1
		
		costFunctionDerivatives = gradientDescent(self.ModelParameters, featureMatrix, labelVector, self.sigmoidFunction)
		
		costFunctionDerivatives = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivatives)
		
		if (self.Regularization) then

			RegularizationDerivatives = self.Regularization:calculateLossFunctionDerivativeRegularizaion(self.ModelParameters, numberOfData)

			costFunctionDerivatives = AqwamMatrixLibrary:add(costFunctionDerivatives, RegularizationDerivatives)

		end

		if (self.Optimizer) then 

			costFunctionDerivatives = self.Optimizer:calculate(costFunctionDerivatives, previousCostFunctionDerivatives) 

		end
		
		previousCostFunctionDerivatives = costFunctionDerivatives

		self.ModelParameters = AqwamMatrixLibrary:add(self.ModelParameters, costFunctionDerivatives)
		
		cost = calculateCost(self.ModelParameters, featureMatrix, labelVector, self.sigmoidFunction)
		
		if (self.Regularization) then 

			regularizationCost = self.Regularization:calculateLossFunctionRegularization(self.ModelParameters, numberOfData)

			cost += regularizationCost

		end
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	if self.Optimizer then self.Optimizer:reset() end
	
	return costArray
	
end

function LogisticRegressionModel:predict(featureMatrix)
	
	local z = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	local hypothesis = sigmoidFunctionList[self.sigmoidFunction](z)
	
	if (hypothesis >= 0.5) then
		
		return 1
		
	else
		
		return 0
		
	end

end

return LogisticRegressionModel
