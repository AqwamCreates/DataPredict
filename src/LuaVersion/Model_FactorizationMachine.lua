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

local FactorizationMachineModel = {}

FactorizationMachineModel.__index = FactorizationMachineModel

setmetatable(FactorizationMachineModel, GradientMethodBaseModel)

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
	
	["BinaryCrossEntropy"] = function (h, y) return 
		
		-((y / h) + (1 - y) / (1 - h)) 
		
	end,
	
	["HingeLoss"] = function (h, y)

		local scale = (((h * y) < 1) and 1) or 0

		return -(y * scale)

	end,

}

function FactorizationMachineModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1]

	local latentWeightVectorMatrix = ModelParameters[2]

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local WeightRegularizer = self.WeightRegularizer
	
	local LatentWeightRegularizer = self.LatentWeightRegularizer

	if (WeightRegularizer) then totalCost = totalCost + WeightRegularizer:calculateCost(weightVector) end
	
	if (LatentWeightRegularizer) then totalCost = totalCost + LatentWeightRegularizer:calculateCost(latentWeightVectorMatrix) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function FactorizationMachineModel:calculateHypothesisVector(featureMatrix, saveAllMatrices)
	
	local ModelParameters = self.ModelParameters or {}
	
	local numberOfFeatures = #featureMatrix[1]
	
	local weightVector = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfFeatures, 1})
	
	local latentWeightVectorMatrix = ModelParameters[2] or self:initializeMatrixBasedOnMode({numberOfFeatures, self.latentFactorCount})

	local linearVector = AqwamTensorLibrary:dotProduct(featureMatrix, weightVector)
	
	local latentVector = AqwamTensorLibrary:dotProduct(featureMatrix, latentWeightVectorMatrix)
	
	local squaredLatentVector = AqwamTensorLibrary:power(latentVector, 2)
	
	local squaredlatentWeightVectorMatrix = AqwamTensorLibrary:power(latentWeightVectorMatrix, 2)
	
	local squaredFeatureMatrix = AqwamTensorLibrary:power(featureMatrix, 2)
	
	local squaredFeatureMatrixDotProductSquaredlatentWeightVectorMatrix = AqwamTensorLibrary:dotProduct(squaredFeatureMatrix, squaredlatentWeightVectorMatrix)
	
	local interactionMatrix = AqwamTensorLibrary:subtract(squaredLatentVector, squaredFeatureMatrixDotProductSquaredlatentWeightVectorMatrix)
	
	local interactionVector = AqwamTensorLibrary:sum(interactionMatrix, 2)
	
	interactionVector = AqwamTensorLibrary:divide(interactionVector, 2)
	
	local zVector = AqwamTensorLibrary:add(linearVector, interactionVector)
	
	local hypothesisVector = AqwamTensorLibrary:applyFunction(binaryFunctionList[self.binaryFunction], zVector)
	
	self.ModelParameters = {weightVector, latentWeightVectorMatrix}
	
	if (saveAllMatrices) then 
		
		self.featureMatrix = featureMatrix 
		
		self.latentVector = latentVector
		
		self.zVector = zVector
		
		self.hypothesisVector = hypothesisVector
		
	end

	return hypothesisVector

end

function FactorizationMachineModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix
	
	local latentVector = self.latentVector
	
	local zVector = self.zVector

	local hypothesisVector = self.hypothesisVector

	if (not featureMatrix) then error("Feature matrix not found.") end
	
	if (not latentVector) then error("Latent vector not found.") end
	
	if (not zVector) then error("Z vector not found.") end
	
	if (not hypothesisVector) then error("Hypothesis vector not found.") end
	
	local ModelParameters = self.ModelParameters or {}
	
	local latentWeightMatrix = ModelParameters[2]
	
	local binaryGradientVector = AqwamTensorLibrary:applyFunction(binaryFunctionGradientList[self.binaryFunction], hypothesisVector, zVector)
	
	lossGradientVector = AqwamTensorLibrary:multiply(lossGradientVector, binaryGradientVector)

	local weightLossFunctionDerivativeVector = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(featureMatrix), lossGradientVector)
	
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
	
	local lossFunctionDerivativeVectorArray = {weightLossFunctionDerivativeVector, latentWeightLossFunctionDerivativeMatrix}

	if (self.areGradientsSaved) then self.lossFunctionDerivativeVectorArray = lossFunctionDerivativeVectorArray end

	return lossFunctionDerivativeVectorArray

end

function FactorizationMachineModel:gradientDescent(lossFunctionDerivativeVectorArray, numberOfData)
	
	local ModelParameters = self.ModelParameters
	
	local weightVector = ModelParameters[1]

	local latentWeightVectorMatrix = ModelParameters[2]
	
	local weightLossFunctionDerivativeVector = lossFunctionDerivativeVectorArray[1]
	
	local latentWeightLossFunctionDerivativeMatrix = lossFunctionDerivativeVectorArray[2]
	
	local WeightRegularizer = self.WeightRegularizer
	
	local LatentWeightRegularizer = self.LatentWeightRegularizer
	
	local WeightOptimizer = self.WeightOptimizer
	
	local LatentWeightOptimizer = self.LatentWeightOptimizer
	
	local weightLearningRate = self.weightLearningRate
	
	local latentWeightLearningRate = self.latentWeightLearningRate

	if (WeightRegularizer) then

		local weightRegularizationDerivatives = WeightRegularizer:calculate(weightVector)

		weightLossFunctionDerivativeVector = AqwamTensorLibrary:add(weightLossFunctionDerivativeVector, weightRegularizationDerivatives)

	end
	
	if (LatentWeightRegularizer) then

		local latentWeightRegularizationDerivatives = LatentWeightRegularizer:calculate(latentWeightVectorMatrix)

		latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:add(latentWeightLossFunctionDerivativeMatrix, latentWeightRegularizationDerivatives)

	end

	weightLossFunctionDerivativeVector = AqwamTensorLibrary:divide(weightLossFunctionDerivativeVector, numberOfData)
	
	latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:divide(latentWeightLossFunctionDerivativeMatrix, numberOfData)

	if (WeightOptimizer) then 

		weightLossFunctionDerivativeVector = WeightOptimizer:calculate(weightLearningRate, weightLossFunctionDerivativeVector, weightVector) 

	else

		weightLossFunctionDerivativeVector = AqwamTensorLibrary:multiply(weightLearningRate, weightLossFunctionDerivativeVector)

	end
	
	if (LatentWeightOptimizer) then 

		latentWeightLossFunctionDerivativeMatrix = LatentWeightOptimizer:calculate(latentWeightLearningRate, latentWeightLossFunctionDerivativeMatrix, latentWeightVectorMatrix) 

	else

		latentWeightLossFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(latentWeightLearningRate, latentWeightLossFunctionDerivativeMatrix)

	end
	
	weightVector = AqwamTensorLibrary:subtract(weightVector, weightLossFunctionDerivativeVector)
	
	latentWeightVectorMatrix = AqwamTensorLibrary:subtract(latentWeightVectorMatrix, latentWeightLossFunctionDerivativeMatrix)

	self.ModelParameters = {weightVector, latentWeightVectorMatrix}

end

function FactorizationMachineModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then 

		self.featureMatrix = nil
		
		self.latentVector = nil
		
		self.zVector = nil
		
		self.hypothesisVector = nil

		self.lossFunctionDerivativeVectorArray = nil

	end

end

function FactorizationMachineModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewFactorizationMachineModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewFactorizationMachineModel, FactorizationMachineModel)
	
	NewFactorizationMachineModel:setName("FactorizationMachine")
	
	local learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewFactorizationMachineModel.weightLearningRate = parameterDictionary.weightLearningRate or learningRate
	
	NewFactorizationMachineModel.latentWeightLearningRate = parameterDictionary.latentWeightLearningRate or learningRate
	
	NewFactorizationMachineModel.latentFactorCount = parameterDictionary.latentFactorCount or defaultLatentFactorCount
	
	NewFactorizationMachineModel.binaryFunction = parameterDictionary.binaryFunction or defaultBinaryFunction

	NewFactorizationMachineModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewFactorizationMachineModel.WeightOptimizer = parameterDictionary.WeightOptimizer
	
	NewFactorizationMachineModel.LatentWeightOptimizer = parameterDictionary.LatentWeightOptimizer

	NewFactorizationMachineModel.WeightRegularizer = parameterDictionary.WeightRegularizer

	NewFactorizationMachineModel.LatentWeightRegularizer = parameterDictionary.LatentWeightRegularizer

	return NewFactorizationMachineModel

end

function FactorizationMachineModel:setWeightOptimizer(WeightOptimizer)

	self.WeightOptimizer = WeightOptimizer

end

function FactorizationMachineModel:setLatentWeightOptimizer(LatentWeightOptimizer)

	self.LatentWeightOptimizer = LatentWeightOptimizer

end

function FactorizationMachineModel:setWeightRegularizer(WeightRegularizer)

	self.WeightRegularizer = WeightRegularizer

end

function FactorizationMachineModel:setLatentWeightRegularizer(LatentWeightRegularizer)

	self.LatentWeightRegularizer = LatentWeightRegularizer

end

function FactorizationMachineModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1]
	
	local latentWeightVectorMatrix = ModelParameters[2]
	
	local numberOfFeatures = #featureMatrix[1]
	
	if (weightVector) then
		
		if (numberOfFeatures ~= #weightVector) then error("The number of features are not the same as the number of weight features.") end
		
	end
	
	if (latentWeightVectorMatrix) then

		if (numberOfFeatures ~= #latentWeightVectorMatrix) then error("The number of features are not the same as the number of latent weight features.") end

	end
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations

	local WeightOptimizer = self.WeightOptimizer
	
	local LatentWeightOptimizer = self.LatentWeightOptimizer

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
	
	if (self.autoResetWeightOptimizers) then
		
		if (WeightOptimizer) then WeightOptimizer:reset() end
		
		if (LatentWeightOptimizer) then LatentWeightOptimizer:reset() end
		
	end

	return costArray

end

function FactorizationMachineModel:predict(featureMatrix)

	return self:calculateHypothesisVector(featureMatrix, false)

end

return FactorizationMachineModel
