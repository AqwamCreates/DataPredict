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

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

local ZTableFunction = require(script.Parent.Parent.Cores.ZTableFunction)

local FactorizationMachineModel = {}

FactorizationMachineModel.__index = FactorizationMachineModel

setmetatable(FactorizationMachineModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultLatentFactorCount = 1

local defaultBinaryFunction = "None"

local defaultCostFunction = "MeanSquaredError"

local epsilon = 1e-14

local epsilonComplement = 1 - epsilon

local function calculateProbabilityDensityFunctionValue(z)

	return (math.exp(-0.5 * math.pow(z, 2)) / math.sqrt(2 * math.pi))

end

local binaryFunctionList = {
	
	["None"] = function (z) return z end,

	["Sigmoid"] = function (z) return 1 / (1 + math.exp(-1 * z)) end,

	["Tanh"] = function (z) return math.tanh(z) end,

	["ReLU"] = function (z) return math.max(0, z) end,

	["LeakyReLU"] = function (z) return math.max((0.01 * z), z) end,

	["ELU"] = function (z) return if (z > 0) then z else (math.exp(z) - 1) end,

	["Gaussian"] = function (z) return math.exp(-math.pow(z, 2)) end,

	["SiLU"] = function (z) return z / (1 + math.exp(-z)) end,

	["Swish"] = function (z) return (z / (1 + math.exp(-z))) end,

	["Mish"] = function (z) return z * math.tanh(math.log(1 + math.exp(z))) end,

	["HardSigmoid"] = function (z)

		local x = (z + 1) / 2

		if (x < 0) then return 0 elseif (x > 1) then return 1 else return x end

	end,

	["BipolarSigmoid"] = function (z) return (2 / (1 + math.exp(-z)) - 1) end,

	["SoftSign"] = function (z) return (z / (1 + math.abs(z))) end,

	["SoftPlus"] = function (z) return (1 + math.exp(z)) end,

	["ArcTangent"] = function (z) return (2 / math.pi) * math.atan(z) end,

	["LogitLink"] = function (z) 

		local x = math.clamp(z, epsilon, epsilonComplement)

		return math.log(x / (1 - x))

	end,

	["LogitInverseLink"] = function (z) return (1 / (1 + math.exp(-1 * z))) end,

	["ProbitLink"] = function (z) return ZTableFunction:getStandardNormalInverseCumulativeDistributionFunctionValue(math.clamp(z, 0, 1)) end,

	["ProbitInverseLink"] = function (z) return ZTableFunction:getStandardNormalCumulativeDistributionFunctionValue(math.clamp(z, -3.9, 3.9)) end,

	["LogLogLink"] = function (z) return math.log(-math.log(math.clamp(z, epsilon, 1))) end,

	["LogLogInverseLink"] = function (z) return math.exp(-math.exp(z)) end,

	["ComplementaryLogLogLink"] = function (z) return math.log(-math.log(1 - math.clamp(z, 0, epsilonComplement))) end,

	["ComplementaryLogLogInverseLink"] = function (z) return (1 - math.exp(-math.exp(z))) end,

	["LogLink"] = function (z) return math.log(math.max(z, epsilon)) end,

	["LogInverseLink"] = function (z) return math.exp(z) end,

	["InverseLink"] = function (z) return (1 / z) end,

	["InverseInverseLink"] = function (z) return z end,

	["SquareRootLink"] = function (z) return math.sqrt(z) end,

	["SquareRootInverseLink"] = function (z) return (1 / math.sqrt(z)) end,

	["SquareInverseLink"] = function (z) return (1 / math.pow(z, 2)) end,

	["SquareInverseInverseLink"] = function (z) return math.pow(z, 2) end,

}

local binaryFunctionGradientList = {
	
	["None"] = function (h, z) return 1 end,

	["Sigmoid"] = function (a, z) return (a * (1 - a)) end,

	["Tanh"] = function (a, z) return (1 - math.pow(a, 2)) end,

	["ReLU"] = function (a, z) if (z > 0) then return 1 else return 0 end end,

	["LeakyReLU"] = function (a, z) if (z > 0) then return 1 else return 0.01 end end,

	["ELU"] = function (a, z) if (z > 0) then return 1 else return math.exp(z) end end,

	["Gaussian"] = function (a, z) return -2 * z * math.exp(-math.pow(z, 2)) end,

	["SiLU"] = function (a, z)

		local sigmoidValue = 1 / (1 + math.exp(-z))

		return (sigmoidValue * (1 + (z * (1 - sigmoidValue))))

	end,

	["Swish"] = function (a, z)

		local sigmoidValue = 1 / (1 + math.exp(-z))

		return (sigmoidValue + (a * (1 - sigmoidValue)))

	end,

	["Mish"] = function (a, z) return math.exp(z) * (math.exp(3 * z) + 4 * math.exp(2 * z) + (6 + 4 * z) * math.exp(z) + 4 * (1 + z)) / math.pow((1 + math.pow((math.exp(z) + 1), 2)), 2) end,

	["HardSigmoid"] = function (a, z) return (((a <= 0 or a >= 1) and 0) or 0.5) end,

	["BipolarSigmoid"] = function (a, z) 

		local sigmoidValue = 1 / (1 + math.exp(-z))

		return (2 * sigmoidValue * (1 - sigmoidValue))

	end,

	["SoftSign"] = function (a, z) return (1 / ((1 + math.abs(z))^2)) end,

	["SoftPlus"] = function (a, z) return 1 / (1 + math.exp(-1 * z)) end,

	["ArcTangent"] = function (a, z) return ((2 / math.pi) * (1 / (1 + z^2))) end,

	["LogitLink"] = function (a, z) 

		local x = math.clamp(z, epsilon, epsilonComplement)

		return (1 / (x * (1 - x))) 

	end,

	["LogitInverseLink"] = function (a, z) return (a * (1 - a)) end,

	["ProbitLink"] = function (a, z) return 1 / calculateProbabilityDensityFunctionValue(a) end,

	["ProbitInverseLink"] = function (a, z) return calculateProbabilityDensityFunctionValue(z) end,

	["LogLogLink"] = function (a, z)

		local x = math.clamp(z, epsilon, 1)

		return x / (x * math.log(x)) 

	end,

	["LogLogInverseLink"] = function (a, z) return -math.exp(z) * math.exp(-math.exp(z)) end,

	["ComplementaryLogLogLink"] = function (a, z)

		local x = math.clamp(z, 0, epsilonComplement)

		return 1 / ((1 - x) * math.log(1 - x)) 

	end,

	["ComplementaryLogLogInverseLink"] = function (a, z) return math.exp(z) * math.exp(-math.exp(z)) end,

	["LogLink"] = function (a, z) return 1 / math.max(z, epsilon) end,

	["LogInverseLink"] = function (a, z) return a end, -- Note: Derivative of exponent(z) is exponent(z), where a = exponent(z). Therefore, we're taking a shortcut to reduce computational resources.

	["InverseLink"] = function (a, z) return (-1 / math.pow(z, 2)) end,

	["InverseInverseLink"] = function (a, z) return 1 end,

	["SquareRootLink"] = function (a, z) return (0.5 / (1.5 * math.sqrt(z))) end,

	["SquareRootInverseLink"] = function (a, z) return (-0.5 / math.pow(z, 1.5)) end,

	["SquareInverseLink"] = function (a, z) return (-2 / math.pow(z, 3)) end,

	["SquareInverseInverseLink"] = function (a, z) return (2 * z) end,

}

local lossFunctionList = {

	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,

	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,
	
	["BinaryCrossEntropy"] = function (h, y) return -((y * math.log(h)) + ((1 - y) * math.log(1 - h))) end,
	
	["HingeLoss"] = function (h, y) return math.max(0, (1 - (h * y))) end,

}

local lossFunctionList = {

	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,

	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,

	["BinaryCrossEntropy"] = function (h, y) return -((y * math.log(h)) + ((1 - y) * math.log(1 - h))) end,

	["HingeLoss"] = function (h, y) return math.max(0, (1 - (h * y))) end,

	["SquaredHingeLoss"] = function (h, y) return math.pow(math.max(0, (1 - (h * y))), 2) end,

}

local lossFunctionGradientList = {

	["MeanSquaredError"] = function (h, y) return (2 * (h - y)) end,

	["MeanAbsoluteError"] = function (h, y) return math.sign(h - y) end,

	["BinaryCrossEntropy"] = function (h, y) return ((h - y) / (h * (1 - h))) end,

	["HingeLoss"] = function (h, y)

		local scale = (((h * y) < 1) and 1) or 0

		return -(y * scale)

	end,

	["SquaredHingeLoss"] = function (h, y)

		local scale = (((h * y) < 1) and 1) or 0

		return -(2 * y * scale)

	end,

}

function FactorizationMachineModel:calculateCost(hypothesisVector, labelVector, hasBias)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1]

	local latentWeightVectorMatrix = ModelParameters[2]

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local WeightRegularizer = self.WeightRegularizer
	
	local LatentWeightRegularizer = self.LatentWeightRegularizer

	if (WeightRegularizer) then totalCost = totalCost + WeightRegularizer:calculateCost(weightVector, hasBias) end
	
	if (LatentWeightRegularizer) then totalCost = totalCost + LatentWeightRegularizer:calculateCost(latentWeightVectorMatrix, hasBias) end

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

function FactorizationMachineModel:gradientDescent(lossFunctionDerivativeVectorArray, numberOfData, hasBias)
	
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

		local weightRegularizationDerivatives = WeightRegularizer:calculate(weightVector, hasBias)

		weightLossFunctionDerivativeVector = AqwamTensorLibrary:add(weightLossFunctionDerivativeVector, weightRegularizationDerivatives)

	end
	
	if (LatentWeightRegularizer) then

		local latentWeightRegularizationDerivatives = LatentWeightRegularizer:calculate(latentWeightVectorMatrix, hasBias)

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

function FactorizationMachineModel:update(lossGradientVector, hasBias, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData, hasBias)

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
	
	local hasBias = self:checkIfFeatureMatrixHasBias(featureMatrix)

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

		local lossGradientVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisVector, labelVector)

		self:update(lossGradientVector, hasBias, true)

	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) or self:checkIfNan(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	if (self.autoResetConvergenceCheck) then self:resetConvergenceCheck() end
	
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
