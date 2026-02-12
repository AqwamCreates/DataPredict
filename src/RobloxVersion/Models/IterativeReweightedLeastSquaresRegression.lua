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

local IterativeMethodBaseModel = require(script.Parent.IterativeMethodBaseModel)

local ZTableFunction = require(script.Parent.Parent.Cores.ZTableFunction)

local IterativeReweightedLeastSquaresRegressionModel = {}

IterativeReweightedLeastSquaresRegressionModel.__index = IterativeReweightedLeastSquaresRegressionModel

setmetatable(IterativeReweightedLeastSquaresRegressionModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLinkFunction = "Linear"

local defaultLearningRate = 1

local defaultPValue = 2

local defaultModelParametersInitializationMode = "Zero"

local cutOffFunction = function (value) return (((value < 0.5) and 0) or 1) end

local function calculateProbabilityDensityFunctionValue(z)

	return (math.exp(-0.5 * math.pow(z, 2)) / math.sqrt(2 * math.pi))

end

local linkFunctionList = {

	["Logistic"] = function (z) return (1/(1 + math.exp(-z))) end,

	["Logit"] = function (z) return (1/(1 + math.exp(-z))) end,

	["Probit"] = function(z) return ZTableFunction:getStandardNormalCumulativeDistributionFunctionValue(math.clamp(z, -3.9, 3.9)) end,

	["LogLog"] = function(z) return math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(z) return (1 - math.exp(-math.exp(z))) end,

}

function IterativeReweightedLeastSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	parameterDictionary.modelParametersInitializationMode = parameterDictionary.modelParametersInitializationMode or defaultModelParametersInitializationMode

	local NewIterativeReweightedLeastSquaresRegressionModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewIterativeReweightedLeastSquaresRegressionModel, IterativeReweightedLeastSquaresRegressionModel)
	
	NewIterativeReweightedLeastSquaresRegressionModel:setName("IterativeReweightedLeastSquaresRegression")
	
	NewIterativeReweightedLeastSquaresRegressionModel.linkFunction = parameterDictionary.linkFunction or defaultLinkFunction
	
	NewIterativeReweightedLeastSquaresRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewIterativeReweightedLeastSquaresRegressionModel.pValue = parameterDictionary.pValue or defaultPValue

	return NewIterativeReweightedLeastSquaresRegressionModel

end

function IterativeReweightedLeastSquaresRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfdata = #featureMatrix

	if (numberOfdata ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local betaVector = self.ModelParameters

	if (betaVector) then

		if (numberOfFeatures ~= #betaVector) then error("The number of features are not the same as the model parameters.") end

	else

		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end
	
	local linkFunction = self.linkFunction
	
	local linkFunctionToApply = linkFunctionList[linkFunction]
	
	if (not linkFunctionToApply) and (linkFunction ~= "Linear") then error("Invalid link function.") end
	
	local learningRate = self.learningRate
	
	local pValue = self.pValue
	
	local weightFunctionToApply = function(labelValue, hypothesisValue) return math.pow(math.abs(labelValue - hypothesisValue), (pValue - 2)) end
	
	local costFunctionToApply = function(labelValue, hypothesisValue) return math.pow(math.abs(labelValue - hypothesisValue), pValue) end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local costArray = {}

	local numberOfIterations = 0
	
	local tansposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
	
	local diagonalMatrix = AqwamTensorLibrary:createTensor({numberOfdata, numberOfdata})
	
	local responseVector

	local errorVector
	
	local targetBetaVector
	
	local betaChangeVector
	
	local costVector
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()
		
		responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)
		
		if (linkFunctionToApply) then responseVector = AqwamTensorLibrary:applyFunction(linkFunctionToApply, responseVector) end
		
		errorVector = AqwamTensorLibrary:applyFunction(weightFunctionToApply, labelVector, responseVector)

		for dataIndex, unwrappedErrorVector in ipairs(errorVector) do diagonalMatrix[dataIndex][dataIndex] = unwrappedErrorVector[1] end
		
		targetBetaVector = AqwamTensorLibrary:dotProduct(tansposedFeatureMatrix, diagonalMatrix, featureMatrix)
		
		targetBetaVector = AqwamTensorLibrary:inverse(targetBetaVector)
		
		targetBetaVector = AqwamTensorLibrary:dotProduct(targetBetaVector, tansposedFeatureMatrix, diagonalMatrix, labelVector)
		
		betaChangeVector = AqwamTensorLibrary:subtract(targetBetaVector, betaVector)
		
		betaChangeVector = AqwamTensorLibrary:multiply(learningRate, betaChangeVector)
		
		betaVector = AqwamTensorLibrary:add(betaVector, betaChangeVector)
		
		costVector = AqwamTensorLibrary:applyFunction(costFunctionToApply, labelVector, responseVector)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return AqwamTensorLibrary:sum(costVector)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	self.ModelParameters = betaVector

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	return costArray

end

function IterativeReweightedLeastSquaresRegressionModel:predict(featureMatrix, returnOriginalOutput)
	
	local linkFunctionToApply = linkFunctionList[self.linkFunction]
	
	local betaVector = self.ModelParameters
	
	if (not betaVector) then
		
		local numberOfFeatures = #featureMatrix[1]
		
		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})
		
		self.ModelParameters = betaVector
		
	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)
	
	if (linkFunctionToApply) then predictedVector = AqwamTensorLibrary:applyFunction(linkFunctionToApply, predictedVector) end
	
	if (linkFunctionToApply) and (not returnOriginalOutput) then return AqwamTensorLibrary:applyFunction(cutOffFunction, predictedVector) end

	return predictedVector

end

return IterativeReweightedLeastSquaresRegressionModel
