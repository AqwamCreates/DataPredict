--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

LinearRegressionModel = {}

LinearRegressionModel.__index = LinearRegressionModel

setmetatable(LinearRegressionModel, GradientMethodBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultLossFunction = "L2"

local lossFunctionList = {

	["L1"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		return AqwamMatrixLibrary:applyFunction(math.abs, part1) 

	end,

	["L2"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		local part2 = AqwamMatrixLibrary:power(part1, 2) 

		return AqwamMatrixLibrary:divide(part2, 2)

	end,

}

function LinearRegressionModel:calculateCost(hypothesisVector, labelVector, numberOfData)
	
	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local costVector = lossFunctionList[self.lossFunction](hypothesisVector, labelVector) 
	
	local totalCost = AqwamMatrixLibrary:sum(costVector)
	
	if (self.Regularization) then
		
		totalCost = self.Regularization:calculateRegularization(self.ModelParameters)
		
	end

	local averageCost = totalCost / numberOfData
	
	return averageCost
	
end

function LinearRegressionModel:calculateHypothesisVector(featureMatrix, saveFeatureMatrix)
	
	local hypothesisVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	if (saveFeatureMatrix) then 
		
		self.featureMatrix = featureMatrix
		
	end

	return hypothesisVector

end

function LinearRegressionModel:calculateCostFunctionDerivativeMatrix(lossMatrix)
	
	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end
	
	local featureMatrix = self.featureMatrix
	
	if (featureMatrix == nil) then error("Feature matrix not found.") end
	
	local costFunctionDerivativeMatrix = AqwamMatrixLibrary:dotProduct(AqwamMatrixLibrary:transpose(featureMatrix), lossMatrix)
	
	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix
	
end

function LinearRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)
	
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

function LinearRegressionModel:update(lossMatrix, clearFeatureMatrix, doNotUpdateModelParameters)
	
	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end
	
	local numberOfData = #lossMatrix
	
	local costFunctionDerivativeMatrix = self:calculateCostFunctionDerivativeMatrix(lossMatrix)
	
	self.ModelParameters = self:gradientDescent(costFunctionDerivativeMatrix, numberOfData)
	
	if (clearFeatureMatrix) then self.featureMatrix = nil end
	
end

function LinearRegressionModel.new(maximumNumberOfIterations, learningRate, lossFunction)
	
	local NewLinearRegressionModel = GradientMethodBaseModel.new()
	
	setmetatable(NewLinearRegressionModel, LinearRegressionModel)
	
	NewLinearRegressionModel.maximumNumberOfIterations = maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	NewLinearRegressionModel.learningRate = learningRate or defaultLearningRate
	
	NewLinearRegressionModel.lossFunction = lossFunction or defaultLossFunction
	
	NewLinearRegressionModel.Optimizer = nil
	
	NewLinearRegressionModel.Regularization = nil
	
	return NewLinearRegressionModel
	
end

function LinearRegressionModel:setParameters(maximumNumberOfIterations, learningRate, lossFunction)

	self.maximumNumberOfIterations = maximumNumberOfIterations or self.maximumNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.lossFunction = lossFunction or self.lossFunction
	
end

function LinearRegressionModel:setOptimizer(Optimizer)
	
	self.Optimizer = Optimizer
	
end

function LinearRegressionModel:setRegularization(Regularization)
	
	self.Regularization = Regularization
	
end

function LinearRegressionModel:train(featureMatrix, labelVector)

	local cost

	local costArray = {}
	
	local numberOfIterations = 0
	
	local numberOfData = #featureMatrix
	
	local lossFunction = self.lossFunction
	
	local Regularization = self.Regularization
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
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
			
			return self:calculateCost(hypothesisVector, labelVector, numberOfData)
			
		end)
		
		if cost then 
			
			table.insert(costArray, cost)
			
			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
		end
		
		local lossVector = AqwamMatrixLibrary:subtract(hypothesisVector, labelVector)
		
		self:update(lossVector, true, false)
		
	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values") end
	
	if (self.Optimizer) and (self.autoResetOptimizers) then self.Optimizer:reset() end
	
	return costArray
	
end

function LinearRegressionModel:predict(featureMatrix)
	
	local predictedVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	if (type(predictedVector) == "number") then predictedVector = {{predictedVector}} end
	
	return predictedVector

end

return LinearRegressionModel