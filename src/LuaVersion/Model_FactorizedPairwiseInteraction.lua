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

local ZTableFunction = require("Core_ZTableFunction")

local FactorizedPairwiseInteractionModel = {}

FactorizedPairwiseInteractionModel.__index = FactorizedPairwiseInteractionModel

setmetatable(FactorizedPairwiseInteractionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultLatentFactorCount = 1

local defaultBinaryFunction = "None"

local defaultCostFunction = "MeanSquaredError"

local function calculateProbabilityDensityFunctionValue(z)

	return (math.exp(-0.5 * math.pow(z, 2)) / math.sqrt(2 * math.pi))

end

local binaryFunctionList = {
	
	["None"] = function (z) return z end,

	["Logistic"] = function (z) return (1/(1 + math.exp(-z))) end,

	["Logit"] = function (z) return (1/(1 + math.exp(-z))) end,

	["Probit"] = function(z) return ZTableFunction:getStandardNormalCumulativeDistributionFunctionValue(math.clamp(z, -3.9, 3.9)) end,

	["LogLog"] = function(z) return math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(z) return (1 - math.exp(-math.exp(z))) end,

	["Tanh"] = function (z) return math.tanh(z) end,

	["HardSigmoid"] = function (z)

		local x = (z + 1) / 2

		if (x < 0) then return 0 elseif (x > 1) then return 1 else return x end

	end,

	["SoftSign"] = function (z) return (z / (1 + math.abs(z))) end,

	["ArcTangent"] = function (z) return (2 / math.pi) * math.atan(z) end,

	["BipolarSigmoid"] = function (z) return (2 / (1 + math.exp(-z)) - 1) end,

}

local binaryFunctionGradientList = {
	
	["None"] = function (h, z) return 1 end,

	["Logistic"] = function (h, z) return (h * (1 - h)) end,

	["Logit"] = function (h, z) return (h * (1 - h)) end,

	["Probit"] = function (h, z) return calculateProbabilityDensityFunctionValue(z) end,

	["LogLog"] = function(h, z) return -math.exp(z) * math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(h, z) return math.exp(z) * math.exp(-math.exp(z)) end,

	["Tanh"] = function (h, z) return (1 - math.pow(h, 2)) end,

	["HardSigmoid"] = function (h, z) return ((h <= 0 or h >= 1) and 0) or 0.5 end,

	["SoftSign"] = function (h, z) return (1 / ((1 + math.abs(z))^2)) end,

	["ArcTangent"] = function (h, z) return ((2 / math.pi) * (1 / (1 + z^2))) end,

	["BipolarSigmoid"] = function (h, z) 

		local sigmoidValue = 1 / (1 + math.exp(-z))

		return (2 * sigmoidValue * (1 - sigmoidValue))

	end,

}

local lossFunctionList = {

	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,

	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,
	
	["BinaryCrossEntropy"] = function (h, y) return -((y * math.log(h)) + ((1 - y) * math.log(1 - h))) end,
	
	["HingeLoss"] = function (h, y) return math.max(0, (1 - (h * y))) end,

}

local lossFunctionGradientList = {

	["MeanSquaredError"] = function (h, y) return (2 * (h - y)) end,

	["MeanAbsoluteError"] = function (h, y) return math.sign(h - y) end,
	
	["BinaryCrossEntropy"] = function (h, y) return ((h - y) / (h * (1 - h))) end,
	
	["HingeLoss"] = function (h, y)

		local scale = (((h * y) < 1) and 1) or 0

		return -(y * scale)

	end,

}

function FactorizedPairwiseInteractionModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1]

	local latentWeightVectorMatrix = ModelParameters[2]

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local Regularizer = self.Regularizer
	
	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(latentWeightVectorMatrix) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function FactorizedPairwiseInteractionModel:calculateHypothesisVector(featureMatrix, saveAllMatrices)
	
	local numberOfFeatures = #featureMatrix[1]
	
	local latentWeightVectorMatrix = self.ModelParameters or self:initializeMatrixBasedOnMode({numberOfFeatures, self.latentFactorCount})
	
	local latentVector = AqwamTensorLibrary:dotProduct(featureMatrix, latentWeightVectorMatrix)
	
	local squaredLatentVector = AqwamTensorLibrary:power(latentVector, 2)
	
	local squaredlatentWeightVectorMatrix = AqwamTensorLibrary:power(latentWeightVectorMatrix, 2)
	
	local squaredFeatureMatrix = AqwamTensorLibrary:power(featureMatrix, 2)
	
	local squaredFeatureMatrixDotProductSquaredlatentWeightVectorMatrix = AqwamTensorLibrary:dotProduct(squaredFeatureMatrix, squaredlatentWeightVectorMatrix)
	
	local interactionMatrix = AqwamTensorLibrary:subtract(squaredLatentVector, squaredFeatureMatrixDotProductSquaredlatentWeightVectorMatrix)
	
	local interactionVector = AqwamTensorLibrary:sum(interactionMatrix, 2)
	
	local zVector = AqwamTensorLibrary:divide(interactionVector, 2)
	
	local hypothesisVector = AqwamTensorLibrary:applyFunction(binaryFunctionList[self.binaryFunction], zVector)
	
	self.ModelParameters = latentWeightVectorMatrix
	
	if (saveAllMatrices) then 
		
		self.featureMatrix = featureMatrix 
		
		self.latentVector = latentVector
		
		self.zVector = zVector
		
		self.hypothesisVector = hypothesisVector
		
	end

	return hypothesisVector

end

function FactorizedPairwiseInteractionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix
	
	local latentVector = self.latentVector
	
	local zVector = self.zVector

	local hypothesisVector = self.hypothesisVector

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	if (not latentVector) then error("Latent vector not found.") end
	
	if (not zVector) then error("Z vector not found.") end
	
	if (not hypothesisVector) then error("Hypothesis vector not found.") end
	
	local latentWeightMatrix = self.ModelParameters or {}
	
	local binaryGradientVector = AqwamTensorLibrary:applyFunction(binaryFunctionGradientList[self.binaryFunction], hypothesisVector, zVector)
	
	lossGradientVector = AqwamTensorLibrary:multiply(lossGradientVector, binaryGradientVector)
	
	local latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:createTensor({#featureMatrix[1], self.latentFactorCount}, 0)
	
	local unwrappedLatentWeightFeatureVector
	
	local unwrappedLatentVector
	
	local unwrappedLatentWeightLossFunctionDerivativeVector
	
	local unwrappedLatentWeightVector
	
	local lossGradientValue
	
	local scaledLossGradientValue
	
	local partialGradientValue
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		lossGradientValue = lossGradientVector[dataIndex][1]

		if (lossGradientValue ~= 0) then
			
			unwrappedLatentVector = latentVector[dataIndex]
			
			for featureIndex, featureValue in ipairs(unwrappedFeatureVector) do
				
				if (featureValue ~= 0) then

					unwrappedLatentWeightLossFunctionDerivativeVector = latentWeightLossFunctionDerivativeMatrix[featureIndex]
					
					unwrappedLatentWeightVector = latentWeightMatrix[featureIndex]
					
					scaledLossGradientValue = lossGradientValue * featureValue
					
					for latentFactorIndex, latentValue in ipairs(unwrappedLatentVector) do

						partialGradientValue = scaledLossGradientValue * (latentValue - (unwrappedLatentWeightVector[latentFactorIndex] * featureValue))

						unwrappedLatentWeightLossFunctionDerivativeVector[latentFactorIndex] = unwrappedLatentWeightLossFunctionDerivativeVector[latentFactorIndex] + partialGradientValue					

					end

					latentWeightLossFunctionDerivativeMatrix[featureIndex] = unwrappedLatentWeightLossFunctionDerivativeVector
					
				end
				
			end
			
		end

	end 

	if (self.areGradientsSaved) then self.latentWeightLossFunctionDerivativeMatrix = latentWeightLossFunctionDerivativeMatrix end

	return latentWeightLossFunctionDerivativeMatrix

end

function FactorizedPairwiseInteractionModel:gradientDescent(latentWeightLossFunctionDerivativeMatrix, numberOfData)
	
	local latentWeightVectorMatrix = self.ModelParameters
	
	local Regularizer = self.Regularizer
	
	local Optimizer = self.Optimizer
	
	local learningRate = self.learningRate
	
	if (Regularizer) then

		local latentWeightRegularizationDerivatives = Regularizer:calculate(latentWeightVectorMatrix)

		latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:add(latentWeightLossFunctionDerivativeMatrix, latentWeightRegularizationDerivatives)

	end
	
	latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:divide(latentWeightLossFunctionDerivativeMatrix, numberOfData)
	
	if (Optimizer) then 

		latentWeightLossFunctionDerivativeMatrix = Optimizer:calculate(learningRate, latentWeightLossFunctionDerivativeMatrix, latentWeightVectorMatrix) 

	else

		latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, latentWeightLossFunctionDerivativeMatrix)

	end
	
	latentWeightVectorMatrix = AqwamTensorLibrary:subtract(latentWeightVectorMatrix, latentWeightLossFunctionDerivativeMatrix)

	self.ModelParameters = latentWeightVectorMatrix

end

function FactorizedPairwiseInteractionModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil
		
		self.latentVector = nil
		
		self.zVector = nil
		
		self.hypothesisVector = nil

		self.latentWeightLossFunctionDerivativeMatrix = nil

	end

end

function FactorizedPairwiseInteractionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewFactorizedPairwiseInteractionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewFactorizedPairwiseInteractionModel, FactorizedPairwiseInteractionModel)
	
	NewFactorizedPairwiseInteractionModel:setName("FactorizedPairwiseInteraction")

	NewFactorizedPairwiseInteractionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewFactorizedPairwiseInteractionModel.latentFactorCount = parameterDictionary.latentFactorCount or defaultLatentFactorCount
	
	NewFactorizedPairwiseInteractionModel.binaryFunction = parameterDictionary.binaryFunction or defaultBinaryFunction

	NewFactorizedPairwiseInteractionModel.costFunction = parameterDictionary.costFunction or defaultCostFunction
	
	NewFactorizedPairwiseInteractionModel.Optimizer = parameterDictionary.Optimizer

	NewFactorizedPairwiseInteractionModel.Regularizer = parameterDictionary.Regularizer

	return NewFactorizedPairwiseInteractionModel

end

function FactorizedPairwiseInteractionModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function FactorizedPairwiseInteractionModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function FactorizedPairwiseInteractionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local latentWeightVectorMatrix = self.ModelParameters
	
	local numberOfFeatures = #featureMatrix[1]
	
	if (latentWeightVectorMatrix) then

		if (numberOfFeatures ~= #latentWeightVectorMatrix) then error("The number of features are not the same as the number of latent weight features.") end

	end
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
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

		local lossGradientVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisVector, labelVector)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	if (self.autoResetWeightOptimizers) and (Optimizer) then Optimizer:reset() end

	return costArray

end

function FactorizedPairwiseInteractionModel:predict(featureMatrix)

	return self:calculateHypothesisVector(featureMatrix, false)

end

return FactorizedPairwiseInteractionModel
