--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_GradientMethodBaseModel")

LogisticRegressionModel = {}

LogisticRegressionModel.__index = LogisticRegressionModel

setmetatable(LogisticRegressionModel, GradientMethodBaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultSigmoidFunction = "Sigmoid"

local sigmoidFunctionList = {

	["Sigmoid"] = function (z) return 1/(1 + math.exp(-1 * z)) end,
	
	["Tanh"] = function (z) return math.tanh(z) end

}

local lossFunctionList = {
	
	["Sigmoid"] = function (y, h) return -(y * math.log(h) + (1 - y) * math.log(1 - h)) end,
	
	["Tanh"] = function (y, h) return (y - h)^2 end
	
}

local cutOffFunctionList = {
	
	["Sigmoid"] = function (x) 

		if (x >= 0.5) then 

			return 1

		else 

			return 0 

		end 

	end,
	
	["Tanh"] = function (x) 

		if (x > 0) then 

			return 1

		elseif (x < 0) then
			
			return -1
			
		else
			
			return 0

		end 

	end
	
}

local function calculateCost(hypothesisVector, labelVector, sigmoidFunction)

	local numberOfData = #labelVector

	local costVector = AqwamMatrixLibrary:applyFunction(lossFunctionList[sigmoidFunction], labelVector, hypothesisVector)

	local totalCost = AqwamMatrixLibrary:sum(costVector)

	local averageCost = totalCost / numberOfData

	return averageCost

end

function LogisticRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)

	local zVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)

	if (saveFeatureMatrix) then 

		self.featureMatrix = featureMatrix

	end
	
	if (type(zVector) == "number") then zVector = {{zVector}} end

	local hypothesisVector = AqwamMatrixLibrary:applyFunction(sigmoidFunctionList[self.sigmoidFunction], zVector)

	return hypothesisVector

end

function LogisticRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)
	
	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local featureMatrix = self.featureMatrix

	if (featureMatrix == nil) then error("Feature matrix not found.") end
	
	local costFunctionDerivativeMatrix = AqwamMatrixLibrary:dotProduct(AqwamMatrixLibrary:transpose(featureMatrix), lossMatrix)
	
	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix

end

function LogisticRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)
	
	if (type(costFunctionDerivativeMatrix) == "number") then costFunctionDerivativeMatrix = {{costFunctionDerivativeMatrix}} end
	
	if (self.Regularization) then

		local regularizationDerivatives = self.Regularization:calculateRegularizationDerivatives(self.ModelParameters)

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:add(costFunctionDerivativeMatrix, regularizationDerivatives)

	end
	
	costFunctionDerivativeMatrix = AqwamMatrixLibrary:divide(costFunctionDerivativeMatrix, numberOfData)

	if (self.Optimizer) then 

		costFunctionDerivativeMatrix = self.Optimizer:calculate(self.learningRate, costFunctionDerivativeMatrix) 

	else

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivativeMatrix)

	end

	local newModelParameters = AqwamMatrixLibrary:subtract(self.ModelParameters, costFunctionDerivativeMatrix)

	return newModelParameters

end

function LogisticRegressionModel:update(lossMatrix, clearFeatureMatrix)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrix = self:calculateCostFunctionDerivativeMatrix(lossMatrix)

	self.ModelParameters = self:gradientDescent(costFunctionDerivativeMatrix, numberOfData)

end

function LogisticRegressionModel.new(maxNumberOfIterations, learningRate, sigmoidFunction)
	
	local NewLogisticRegressionModel = GradientMethodBaseModel.new()

	setmetatable(NewLogisticRegressionModel, LogisticRegressionModel)

	NewLogisticRegressionModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewLogisticRegressionModel.learningRate = learningRate or defaultLearningRate

	NewLogisticRegressionModel.sigmoidFunction = sigmoidFunction or defaultSigmoidFunction
	
	NewLogisticRegressionModel.Optimizer = nil
	
	NewLogisticRegressionModel.Regularization = nil
	
	return NewLogisticRegressionModel
	
end

function LogisticRegressionModel:setParameters(maxNumberOfIterations, learningRate, sigmoidFunction)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.sigmoidFunction = sigmoidFunction or self.sigmoidFunction

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
	
	local numberOfData = #featureMatrix
	
	local sigmoidFunction = self.sigmoidFunction
	
	local Regularization = self.Regularization
	
	local maxNumberOfIterations = self.maxNumberOfIterations
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end

	if (self.ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], 1)

	end
	
	repeat
		
		numberOfIterations += 1

		self:iterationWait()

		local hypothesisVector = self:calculateHypothesisVector(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			cost = calculateCost(hypothesisVector, labelVector, sigmoidFunction)

			if (not Regularization) then return cost end

			local regularizationCost = Regularization:calculateRegularization(self.ModelParameters, numberOfData)

			cost += regularizationCost

			return cost

		end)

		if cost then 

			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)

		end

		local lossVector = AqwamMatrixLibrary:subtract(hypothesisVector, labelVector)

		self:update(lossVector, true, false)
		
	until (numberOfIterations == maxNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	if (self.Optimizer) and (self.autoResetOptimizers) then self.Optimizer:reset() end
	
	return costArray
	
end

function LogisticRegressionModel:predict(featureMatrix, returnOriginalOutput)
	
	local outputVector = self:calculateHypothesisVector(featureMatrix, false)
	
	if (returnOriginalOutput == true) then return outputVector end
	
	local cutOffFunction = cutOffFunctionList[self.sigmoidFunction]
	
	local predictedLabelVector = AqwamMatrixLibrary:applyFunction(cutOffFunction, outputVector)
	
	return predictedLabelVector, outputVector

end

return LogisticRegressionModel
