--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local GradientMethodBaseModel = require("Model_GradientMethodBaseModel")

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
	
	local costFunctionDerivatives
	
	local numberOfData = #featureMatrix
	
	local lambda
	
	local regularizationCost
	
	local regularizationDerivatives
	
	local calculatedLearningRate = self.learningRate / numberOfData
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end

	if (self.ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], 1)

	end
	
	repeat
		
		numberOfIterations += 1
		
		self:iterationWait()
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			cost = calculateCost(self.ModelParameters, featureMatrix, labelVector, self.sigmoidFunction)

			if (not self.Regularization) then return cost end

			regularizationCost = self.Regularization:calculateRegularization(self.ModelParameters, numberOfData)

			cost += regularizationCost

			return cost

		end)

		if cost then 

			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)

		end
		
		costFunctionDerivatives = gradientDescent(self.ModelParameters, featureMatrix, labelVector, self.sigmoidFunction)
		
		if (self.Regularization) then

			regularizationDerivatives = self.Regularization:calculateRegularizationDerivatives(self.ModelParameters, numberOfData)

			costFunctionDerivatives = AqwamMatrixLibrary:add(costFunctionDerivatives, regularizationDerivatives)

		end
		
		if (self.Optimizer) then 

			costFunctionDerivatives = self.Optimizer:calculate(calculatedLearningRate, costFunctionDerivatives)
			
		else
			
			costFunctionDerivatives = AqwamMatrixLibrary:multiply(calculatedLearningRate, costFunctionDerivatives)

		end
		
		if (self.areGradientsSaved) then self.Gradients = costFunctionDerivatives end

		self.ModelParameters = AqwamMatrixLibrary:subtract(self.ModelParameters, costFunctionDerivatives)
		
	until (numberOfIterations == self.maxNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	if (self.Optimizer) and (self.AutoResetOptimizers) then self.Optimizer:reset() end
	
	return costArray
	
end

function LogisticRegressionModel:predict(featureMatrix, returnOriginalOutput)
	
	local z = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	local sigmoidFunction = sigmoidFunctionList[self.sigmoidFunction]
	
	if (typeof(z) == "number") then z = {{z}} end
	
	local outputVector = AqwamMatrixLibrary:applyFunction(sigmoidFunction, z)
	
	if (returnOriginalOutput == true) then return outputVector end
	
	local cutOffFunction = cutOffFunctionList[self.sigmoidFunction]
	
	local predictedLabelVector = AqwamMatrixLibrary:applyFunction(cutOffFunction, outputVector)
	
	return predictedLabelVector, outputVector

end

return LogisticRegressionModel
