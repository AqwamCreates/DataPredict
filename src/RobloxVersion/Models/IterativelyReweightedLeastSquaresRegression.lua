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

local IterativelyReweightedLeastSquaresRegressionModel = {}

IterativelyReweightedLeastSquaresRegressionModel.__index = IterativelyReweightedLeastSquaresRegressionModel

setmetatable(IterativelyReweightedLeastSquaresRegressionModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLinkFunction = "Linear"

local defaultPValue = 2

local defaultModelParametersInitializationMode = "Diagonal"

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

local linkFunctionGradientList = {

	["Logistic"] = function (h, z) return (h * (1 - h)) end,

	["Logit"] = function (h, z) return (h * (1 - h)) end,

	["Probit"] = function (h, z) return calculateProbabilityDensityFunctionValue(z) end,

	["LogLog"] = function(h, z) return -math.exp(z) * math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(h, z) return math.exp(z) * math.exp(-math.exp(z)) end,

}

function IterativelyReweightedLeastSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	parameterDictionary.modelParametersInitializationMode = parameterDictionary.modelParametersInitializationMode or defaultModelParametersInitializationMode

	local NewIterativelyReweightedLeastSquaresRegressionModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewIterativelyReweightedLeastSquaresRegressionModel, IterativelyReweightedLeastSquaresRegressionModel)
	
	NewIterativelyReweightedLeastSquaresRegressionModel:setName("IterativelyReweightedLeastSquaresRegression")
	
	NewIterativelyReweightedLeastSquaresRegressionModel.linkFunction = parameterDictionary.linkFunction or defaultLinkFunction

	NewIterativelyReweightedLeastSquaresRegressionModel.pValue = parameterDictionary.pValue or defaultPValue

	return NewIterativelyReweightedLeastSquaresRegressionModel

end

function IterativelyReweightedLeastSquaresRegressionModel:train(featureMatrix, labelVector)
	
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
	
	local linkFunctionGradientToApply = linkFunctionGradientList[linkFunction]
	
	local pValue = self.pValue
	
	local weightFunctionToApply = function(labelValue, hypothesisValue) return math.pow(math.abs(labelValue - hypothesisValue), (pValue - 2)) end
	
	local costFunctionToApply = function(labelValue, hypothesisValue) return math.pow(math.abs(labelValue - hypothesisValue), pValue) end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local costArray = {}

	local numberOfIterations = 0
	
	local tansposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
	
	local covarianceMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfdata, numberOfdata}, 1)
	
	local varianceVector
	
	local betaVector
	
	local hypothesisVector
	
	local gradientVector
	
	local costVector
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()
		
		betaVector = AqwamTensorLibrary:dotProduct(tansposedFeatureMatrix, covarianceMatrix, featureMatrix)
		
		betaVector = AqwamTensorLibrary:inverse(betaVector)
		
		betaVector = AqwamTensorLibrary:dotProduct(tansposedFeatureMatrix, covarianceMatrix, labelVector)
		
		hypothesisVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)
		
		if (linkFunctionToApply) then 
			
			hypothesisVector = AqwamTensorLibrary:applyFunction(linkFunctionToApply, hypothesisVector)
			
			gradientVector = AqwamTensorLibrary:applyFunction(linkFunctionGradientToApply, hypothesisVector)
			
		end
		
		varianceVector = AqwamTensorLibrary:applyFunction(weightFunctionToApply, labelVector, hypothesisVector)
		
		if (linkFunctionGradientToApply) then varianceVector = AqwamTensorLibrary:multiply(varianceVector, gradientVector) end
		
		covarianceMatrix = AqwamTensorLibrary:dotProduct(varianceVector, AqwamTensorLibrary:transpose(varianceVector))
		
		costVector = AqwamTensorLibrary:applyFunction(costFunctionToApply, labelVector, hypothesisVector)

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

function IterativelyReweightedLeastSquaresRegressionModel:predict(featureMatrix, returnOriginalOutput)
	
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

return IterativelyReweightedLeastSquaresRegressionModel
