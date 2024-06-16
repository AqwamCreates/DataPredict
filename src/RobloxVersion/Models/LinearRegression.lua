local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

LinearRegressionModel = {}

LinearRegressionModel.__index = LinearRegressionModel

setmetatable(LinearRegressionModel, GradientMethodBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultLossFunction = "L2"

local lossFunctionList = {

	["L1"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["L2"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local distance = AqwamMatrixLibrary:sum(part2)

		return distance 

	end,

}

local function calculateCost(hypothesisVector, labelVector, lossFunction)
	
	local numberOfData = #labelVector
	
	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local costVector = lossFunctionList[lossFunction](hypothesisVector, labelVector) 
	
	local averageCost = costVector / (2 * numberOfData)
	
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
	
	local featureMatrix = self.featureMatrix
	
	if (featureMatrix == nil) then error("Feature matrix not found.") end
	
	local costFunctionDerivativeMatrix = AqwamMatrixLibrary:dotProduct(AqwamMatrixLibrary:transpose(featureMatrix), lossMatrix)
	
	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrix end

	return costFunctionDerivativeMatrix
	
end

function LinearRegressionModel:gradientDescent(costFunctionDerivativeMatrix, numberOfData)
	
	local calculatedLearningRate = self.learningRate / numberOfData

	if (self.Optimizer) then 

		costFunctionDerivativeMatrix = self.Optimizer:calculate(calculatedLearningRate, costFunctionDerivativeMatrix) 

	else

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:multiply(calculatedLearningRate, costFunctionDerivativeMatrix)

	end

	if (self.Regularization) then

		local regularizationDerivatives = self.Regularization:calculateRegularizationDerivatives(self.ModelParameters, numberOfData)

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:add(costFunctionDerivativeMatrix, regularizationDerivatives)

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

function LinearRegressionModel.new(maxNumberOfIterations, learningRate, lossFunction)
	
	local NewLinearRegressionModel = GradientMethodBaseModel.new()
	
	setmetatable(NewLinearRegressionModel, LinearRegressionModel)
	
	NewLinearRegressionModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewLinearRegressionModel.learningRate = learningRate or defaultLearningRate
	
	NewLinearRegressionModel.lossFunction = lossFunction or defaultLossFunction
	
	NewLinearRegressionModel.Optimizer = nil
	
	NewLinearRegressionModel.Regularization = nil
	
	return NewLinearRegressionModel
	
end

function LinearRegressionModel:setParameters(maxNumberOfIterations, learningRate, lossFunction)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

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
	
	local numberOfData = #featureMatrix[1]
	
	local lossFunction = self.lossFunction
	
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
			
			cost = calculateCost(hypothesisVector, labelVector, lossFunction)
			
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
