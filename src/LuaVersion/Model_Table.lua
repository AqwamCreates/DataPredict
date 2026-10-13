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

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

local ZTableFunction = require("Core_ZTableFunction")

local TableModel = {}

TableModel.__index = TableModel

setmetatable(TableModel, IterativeMethodBaseModel)

local defaultLearningRate = 0.1

local defaultCostFunction = "MeanSquaredError"

local defaultActivationFunction = "None"

local epsilon = 1e-14

local epsilonComplement = 1 - epsilon

local function calculateProbabilityDensityFunctionValue(z)

	return (math.exp(-0.5 * math.pow(z, 2)) / math.sqrt(2 * math.pi))

end

local elementWiseActivationFunctionList = {

	["Exponent"] = function (z) return math.exp(z) end,

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

	["SoftPlus"] = function (z) return math.log(1 + math.exp(z)) end,

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

	["Complementary_LogLogLink"] = function (z) return math.log(-math.log(1 - math.clamp(z, 0, epsilonComplement))) end,

	["Complementary_LogLogInverseLink"] = function (z) return (1 - math.exp(-math.exp(z))) end,

	["LogLink"] = function (z) return math.log(math.max(z, epsilon)) end,

	["LogInverseLink"] = function (z) return math.exp(z) end,

	["InverseLink"] = function (z) return (1 / z) end,

	["InverseInverseLink"] = function (z) return z end,

	["SquareRootLink"] = function (z) return math.sqrt(z) end,

	["SquareRootInverseLink"] = function (z) return (1 / math.sqrt(z)) end,

	["SquareInverseLink"] = function (z) return (1 / math.pow(z, 2)) end,

	["SquareInverseInverseLink"] = function (z) return math.pow(z, 2) end,

}

local activationFunctionList = {

	["BinaryStep"] = function (zMatrix)

		local aMatrix = AqwamTensorLibrary:applyFunction(function (z) return (((z <= 0) and 0) or 1) end, zMatrix)

		return aMatrix

	end,

	["Softmax"] = function (zMatrix)

		local exponentZMatrix = AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

		local exponentZSumMatrix = AqwamTensorLibrary:sum(exponentZMatrix, 2)

		local aMatrix = AqwamTensorLibrary:divide(exponentZMatrix, exponentZSumMatrix)

		return aMatrix

	end,

	["StableSoftmax"] = function (zMatrix)

		local maximumZVector = AqwamTensorLibrary:findMaximumValue(zMatrix, 2)

		zMatrix = AqwamTensorLibrary:subtract(zMatrix, maximumZVector)

		local exponentZMatrix = AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

		local exponentZSumMatrix = AqwamTensorLibrary:sum(exponentZMatrix, 2)

		local aMatrix = AqwamTensorLibrary:divide(exponentZMatrix, exponentZSumMatrix)

		return aMatrix

	end,

	["Maxout"] = function (zMatrix)

		local numberOfFeatures = #zMatrix[1]

		local aMatrix = {}

		local unwrappedAVector

		local maximumValue

		local indexWithTheMaximumValue

		for dataIndex, unwrappedZVector in ipairs(zMatrix) do

			unwrappedAVector = table.create(numberOfFeatures, 0)

			maximumValue = -math.huge

			indexWithTheMaximumValue = nil

			for i, zValue in ipairs(unwrappedZVector) do

				if (zValue > maximumValue) then

					maximumValue = zValue

					indexWithTheMaximumValue = i

				end

			end

			unwrappedAVector[indexWithTheMaximumValue] = maximumValue

			aMatrix[dataIndex] = unwrappedAVector

		end

		return aMatrix

	end,

	["None"] = function (zMatrix) return zMatrix end,

}

local elementWiseActivationFunctionDerivativeList = {

	["Exponent"] = function (a, z) return a end,

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

		return 1 / (x * math.log(x)) 

	end,

	["LogLogInverseLink"] = function (a, z) return -math.exp(z) * math.exp(-math.exp(z)) end,

	["Complementary_LogLogLink"] = function (a, z)

		local x = math.clamp(z, 0, epsilonComplement)

		return 1 / ((1 - x) * math.log(1 - x)) 

	end,

	["Complementary_LogLogInverseLink"] = function (a, z) return math.exp(z) * math.exp(-math.exp(z)) end,

	["LogLink"] = function (a, z) return 1 / math.max(z, epsilon) end,

	["LogInverseLink"] = function (a, z) return a end,

	["InverseLink"] = function (a, z) return (-1 / math.pow(z, 2)) end,

	["InverseInverseLink"] = function (a, z) return 1 end,

	["SquareRootLink"] = function (a, z) return (1 / (2 * math.sqrt(z))) end,

	["SquareRootInverseLink"] = function (a, z) return (-0.5 / math.pow(z, 1.5)) end,

	["SquareInverseLink"] = function (a, z) return (-2 / math.pow(z, 3)) end,

	["SquareInverseInverseLink"] = function (a, z) return (2 * z) end,

}

local activationFunctionDerivativeList = {

	["BinaryStep"] = function (aMatrix, zMatrix) return AqwamTensorLibrary:createTensor({#zMatrix, #zMatrix[1]}, 0) end,

	["Softmax"] = function (unwrappedAVector, unwrappedZVector)

		local unwrappedDerivativeVector = table.create(#unwrappedZVector, 0)

		local derivativeValue

		local indicatorValue

		for i, a1Value in ipairs(unwrappedAVector) do

			for j, a2Value in ipairs(unwrappedAVector) do

				indicatorValue = ((i == j) and 1) or 0 

				derivativeValue = a1Value * (indicatorValue - a2Value)

				unwrappedDerivativeVector[i] = unwrappedDerivativeVector[i] + derivativeValue

			end

		end

		return unwrappedDerivativeVector

	end,

	["StableSoftmax"] = function (unwrappedAVector, unwrappedZVector)

		local unwrappedDerivativeVector = table.create(#unwrappedZVector, 0)

		local derivativeValue

		local indicatorValue

		for i, a1Value in ipairs(unwrappedAVector) do

			for j, a2Value in ipairs(unwrappedAVector) do

				indicatorValue = ((i == j) and 1) or 0 

				derivativeValue = a1Value * (indicatorValue - a2Value)

				unwrappedDerivativeVector[i] = unwrappedDerivativeVector[i] + derivativeValue

			end

		end

		return unwrappedDerivativeVector

	end,

	["Maxout"] = function (unwrappedAVector, unwrappedZVector)

		local unwrappedDerivativeVector = table.create(#unwrappedZVector, 0)

		local maximumValue = -math.huge

		local indexWithTheMaximumValue = nil

		for i, zValue in ipairs(unwrappedZVector) do

			if (zValue > maximumValue) then

				maximumValue = zValue

				indexWithTheMaximumValue = i

			end

		end

		unwrappedDerivativeVector[indexWithTheMaximumValue] = 1

		return unwrappedDerivativeVector

	end,

	["None"] = function (unwrappedAVector, unwrappedZVector) return table.create(#unwrappedZVector, 1) end,

}

local lossFunctionList = {

	["MeanAbsoluteError"] = function(generatedLabelValue, labelValue) return math.abs(generatedLabelValue - labelValue) end,

	["MeanSquaredError"] = function(generatedLabelValue, labelValue) return math.pow((generatedLabelValue - labelValue), 2) end,

	["MeanPoissonDeviance"] = function (h, y) return (2 * (y * math.log(y / h) - y + h)) end,

	["MeanGammaDeviance"] = function (h, y) 

		local ratio = y / h

		return (2 * (math.log(ratio) + ratio - 1))

	end,

	["HingeLoss"] = function(generatedLabelValue, labelValue) return math.max(0, (1 - (generatedLabelValue * labelValue))) end,

	["SquaredHingeLoss"] = function (h, y) return math.pow(math.max(0, (1 - (h * y))), 2) end,

	["PerceptronLoss"] = function (h, y) return math.max(0, -(h * y)) end,

	["BinaryCrossEntropy"] = function(generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue) + (1 - labelValue) * math.log(1 - generatedLabelValue)) end,

	["CategoricalCrossEntropy"] = function(generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue)) end,

}

local lossFunctionGradientList = {

	["MeanAbsoluteError"] = function(generatedLabelValue, labelValue) return math.sign(generatedLabelValue - labelValue) end,

	["MeanSquaredError"] = function(generatedLabelValue, labelValue) return (2 * (generatedLabelValue - labelValue)) end,

	["MeanPoissonDeviance"] = function (h, y) return (2 * (1 - (y / h))) end,

	["MeanGammaDeviance"] = function (h, y) return (2 * ((h - y) / math.pow(h, 2))) end,

	["HingeLoss"] = function (generatedLabelValue, labelValue)

		local scale = (((generatedLabelValue * labelValue) < 1) and 1) or 0

		return -(labelValue * scale) 

	end,

	["SquaredHingeLoss"] = function (h, y)

		local margin = 1 - (h * y)

		local scale = ((margin > 0) and margin) or 0

		return -(2 * y * scale)

	end,

	["PerceptronLoss"] = function (h, y) 

		local value = (h * y)

		if (value > 0) then return 0 end

		return -value

	end,

	["BinaryCrossEntropy"] = function (generatedLabelValue, labelValue) return ((generatedLabelValue - labelValue) / (generatedLabelValue * (1 - generatedLabelValue))) end,

	["CategoricalCrossEntropy"] = function (generatedLabelValue, labelValue) return -(labelValue / generatedLabelValue) end,

}

local function activateLayer(zMatrix, activationFunctionName)

	local numberOfData = #zMatrix

	local numberOfFeatures = #zMatrix[1]

	local activationFunction = elementWiseActivationFunctionList[activationFunctionName] 

	local activationMatrix = {}

	if (activationFunction) then

		for dataIndex, unwrappedLayerZVector in ipairs(zMatrix) do

			local unwrappedActivationVector = {}

			for featureIndex = 1, numberOfFeatures, 1 do

				unwrappedActivationVector[featureIndex] = activationFunction(unwrappedLayerZVector[featureIndex])

			end

			activationMatrix[dataIndex] = unwrappedActivationVector

		end

	else

		activationFunction = activationFunctionList[activationFunctionName]

		activationMatrix = activationFunction(zMatrix)

	end

	return activationMatrix

end

local function deriveLayer(activationMatrix, zMatrix, activationFunctionName)

	local numberOfData = #zMatrix

	local numberOfFeatures = #zMatrix[1]

	local activationFunctionDerivativeFunction = elementWiseActivationFunctionDerivativeList[activationFunctionName] 

	local derivativeMatrix = {}

	if (activationFunctionDerivativeFunction) then

		for dataIndex, unwrappedActivationVector in ipairs(activationMatrix) do

			local unwrappedZVector = zMatrix[dataIndex]

			local unwrappedDerivativeVector = {}

			for featureIndex = 1, numberOfFeatures, 1 do

				unwrappedDerivativeVector[featureIndex] = activationFunctionDerivativeFunction(unwrappedActivationVector[featureIndex], unwrappedZVector[featureIndex])

			end

			derivativeMatrix[dataIndex] = unwrappedDerivativeVector

		end

	else

		activationFunctionDerivativeFunction = activationFunctionDerivativeList[activationFunctionName]

		for dataIndex, unwrappedActivationVector in ipairs(activationMatrix) do

			local unwrappedZVector = zMatrix[dataIndex]

			derivativeMatrix[dataIndex] = activationFunctionDerivativeFunction(unwrappedActivationVector, unwrappedZVector)

		end

	end

	return derivativeMatrix

end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

local function createClassesList(labelVector)

	local classesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(classesList, value) then

			table.insert(classesList, value)

		end

	end

	return classesList

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if (not table.find(ClassesList, labelVector[i][1])) then return true end

	end

	return false

end

function TableModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewTableModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewTableModel, TableModel)

	NewTableModel:setName("Table")

	NewTableModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewTableModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewTableModel.activationFunction = parameterDictionary.activationFunction or defaultActivationFunction

	NewTableModel.Optimizer = parameterDictionary.Optimizer

	NewTableModel.Regularizer = parameterDictionary.Regularizer

	NewTableModel.FeaturesList = parameterDictionary.FeaturesList or {}

	NewTableModel.ClassesList = parameterDictionary.ClassesList or {}

	NewTableModel.ModelParameters = parameterDictionary.ModelParameters

	return NewTableModel

end

function TableModel:setLearningRate(learningRate)

	self.learningRate = learningRate

end

function TableModel:getLearningRate()

	return self.learningRate

end

function TableModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function TableModel:getOptimizer()

	return self.Optimizer

end

function TableModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function TableModel:getRegularizer()

	return self.Regularizer

end

function TableModel:generateModelParameters()

	self.ModelParameters = self:initializeMatrixBasedOnMode({#self.FeaturesList, #self.ClassesList})

end

function TableModel:getOutputMatrix(featureVector, saveFeatureIndexArray)

	if (type(featureVector) ~= "table") then featureVector = {{featureVector}} end

	local FeaturesList = self.FeaturesList

	local ClassesList = self.ClassesList

	local ModelParameters = self.ModelParameters

	local featureIndexArray = {}

	if (not ModelParameters) then

		ModelParameters = self:initializeMatrixBasedOnMode({#FeaturesList, #ClassesList})

		self.ModelParameters = ModelParameters

	end

	local zMatrix = {}

	local feature

	local featureIndex

	for i, unwrappedFeatureVector in ipairs(featureVector) do

		feature = unwrappedFeatureVector[1]

		featureIndex = table.find(FeaturesList, feature)

		if (not featureIndex) then error("Feature \"" .. feature ..  "\" does not exist in the features list.") end

		zMatrix[i] = table.clone(ModelParameters[featureIndex])

		featureIndexArray[i] = featureIndex

	end

	if (saveFeatureIndexArray) then self.featureIndexArray = featureIndexArray end

	return zMatrix, featureIndexArray

end

function TableModel:calculateLossFunctionDerivativeMatrix(featureIndexArray, lossGradientMatrix)

	local costFunctionDerivativeMatrix = AqwamTensorLibrary:createTensor({#self.FeaturesList, #self.ClassesList})

	for dataIndex, featureIndex in ipairs(featureIndexArray) do

		costFunctionDerivativeMatrix[featureIndex] = AqwamTensorLibrary:add({costFunctionDerivativeMatrix[featureIndex]}, {lossGradientMatrix[dataIndex]})[1]

	end

	return costFunctionDerivativeMatrix

end

function TableModel:gradientDescent(costFunctionDerivativeMatrix)

	local learningRate = self.learningRate

	local Optimizer = self.Optimizer

	local Regularizer = self.Regularizer

	if (Regularizer) then

		local regularizationDerivativeMatrix = Regularizer:calculate(self.ModelParameters, false)

		costFunctionDerivativeMatrix = AqwamTensorLibrary:add(costFunctionDerivativeMatrix, regularizationDerivativeMatrix)

	end

	if (Optimizer) then

		costFunctionDerivativeMatrix = Optimizer:calculate(learningRate, costFunctionDerivativeMatrix)

	else

		costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrix)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(self.ModelParameters, costFunctionDerivativeMatrix)

end

function TableModel:update(lossGradientMatrix, clearFeatureIndexArray)

	if (type(lossGradientMatrix) ~= "table") then lossGradientMatrix = {{lossGradientMatrix}} end

	local costFunctionDerivativeMatrix = self:calculateLossFunctionDerivativeMatrix(self.featureIndexArray, lossGradientMatrix)

	self:gradientDescent(costFunctionDerivativeMatrix)

	if (clearFeatureIndexArray) then self.featureIndexArray = nil end

end

function TableModel:convertLabelVectorToLogisticMatrix(labelVector)

	if (typeof(labelVector) == "number") then labelVector = {{labelVector}} end

	local ClassesList = self.ClassesList

	local numberOfData = #labelVector

	local numberOfClasses = #ClassesList

	local logisticMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClasses}, 0)

	local label

	local labelPosition

	for data = 1, numberOfData, 1 do

		label = labelVector[data][1]

		labelPosition = table.find(ClassesList, label)

		if (labelPosition) then

			logisticMatrix[data][labelPosition] = 1

		end

	end

	return logisticMatrix

end

function TableModel:processLabelVector(labelVector)

	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(ClassesList)

		if (areNumbersOnly) then table.sort(ClassesList, function(a,b) return a < b end) end

		self.ClassesList = ClassesList

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)

	return logisticMatrix

end

function TableModel:train(featureVector, labelVector)

	local numberOfData = #featureVector

	if (numberOfData ~= #labelVector) then error("Number of rows of feature vector and the label vector is not the same.") end

	local numberOfClasses = #self.ClassesList

	local logisticMatrix

	if (#labelVector[1] == 1) and (numberOfClasses ~= 1) then

		logisticMatrix = self:processLabelVector(labelVector)

	else

		if (#labelVector[1] ~= numberOfClasses) then error("The number of columns for the label matrix is not equal to number of neurons at final layer.") end

		logisticMatrix = labelVector

	end

	local lossFunctionToApply = lossFunctionList[self.costFunction]

	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	local costArray = {}

	local numberOfIterations = 0

	local zMatrix

	local activationMatrix

	local lossGradientMatrix

	local derivativeMatrix

	local layerCostMatrix

	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		zMatrix = self:getOutputMatrix(featureVector, true)

		activationMatrix = activateLayer(zMatrix, self.activationFunction)

		lossGradientMatrix = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, activationMatrix, logisticMatrix)

		derivativeMatrix = deriveLayer(activationMatrix, zMatrix, self.activationFunction)

		layerCostMatrix = AqwamTensorLibrary:multiply(lossGradientMatrix, derivativeMatrix)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			local costVector = AqwamTensorLibrary:applyFunction(lossFunctionToApply, activationMatrix, logisticMatrix)

			local totalCost = AqwamTensorLibrary:sum(costVector)

			local Regularizer = self.Regularizer

			if (Regularizer) then 

				totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters, false) 

			end

			return (totalCost / numberOfData)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		self:update(layerCostMatrix, true)

	until (numberOfIterations == self.maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged. Please repeat the experiment again or change the argument values.") end

	if (self.autoResetOptimizers) then

		for i, Optimizer in ipairs(self.OptimizerArray) do

			if (Optimizer ~= 0) then Optimizer:reset() end

		end

	end

	return costArray

end

function TableModel:predict(featureVector, returnOriginalOutput)

	local zMatrix = self:getOutputMatrix(featureVector, false)

	local activationMatrix = activateLayer(zMatrix, self.activationFunction)

	if (returnOriginalOutput) then return activationMatrix end

	local ClassesList = self.ClassesList

	local outputVector = {}

	local maximumValueVector = {}

	local maximumValue

	local highestClassIndex

	local class

	for i, unwrappedOutputVector in ipairs(activationMatrix) do

		maximumValue = -math.huge

		highestClassIndex = nil

		for classIndex, value in ipairs(unwrappedOutputVector) do

			if (value > maximumValue) then

				maximumValue = value

				highestClassIndex = classIndex

			end

		end

		class = ClassesList[highestClassIndex]

		outputVector[i] = {class}

		maximumValueVector[i] = {maximumValue}

	end

	return outputVector, maximumValueVector

end

function TableModel:setFeaturesList(FeaturesList)

	self.FeaturesList = FeaturesList

end

function TableModel:getFeaturesList()

	return self.FeaturesList

end

function TableModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function TableModel:getClassesList()

	return self.ClassesList

end

return TableModel
