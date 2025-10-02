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

NeuralNetworkModel = {}

NeuralNetworkModel.__index = NeuralNetworkModel

setmetatable(NeuralNetworkModel, GradientMethodBaseModel)

local defaultCostFunction = "MeanSquaredError"

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultActivationFunction = "LeakyReLU"

local defaultDropoutRate = 0

local layerPropertyValueTypeCheckingFunctionList = {

	["NumberOfNeurons"] = function(value)

		local valueType = type(value)

		if (valueType ~= "nil") and (valueType ~= "number") then error("Invalid input for number of neurons!") end 

	end,

	["HasBias"] = function(value)

		local valueType = type(value)

		if (valueType ~= "nil") and (valueType ~= "boolean") then error("Invalid input for has bias!") end 

	end,

	["ActivationFunction"] = function(value)

		local valueType = type(value)

		if (valueType ~= "nil") and (valueType ~= "string") then error("Invalid input for activation function!") end

	end,

	["LearningRate"] = function(value)

		local valueType = type(value)

		if (valueType ~= "nil") and (valueType ~= "number") then error("Invalid input for learning rate!") end

	end,

	["DropoutRate"] = function(value)

		local valueType = type(value)

		if (valueType ~= "nil") and (valueType ~= "number") then error("Invalid input for dropout rate!") end

	end,


}

local costFunctionList = {
	
	["MeanSquaredError"] = function(generatedLabelMatrix, labelMatrix)
		
		local functionToApply = function (generatedLabelValue, labelValue) return math.pow((generatedLabelValue - labelValue), 2) end

		local squaredErrorTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelMatrix, labelMatrix)

		local sumSquaredErrorValue = AqwamTensorLibrary:sum(squaredErrorTensor)

		return sumSquaredErrorValue
		
	end,
	
	["MeanAbsoluteError"] = function(generatedLabelMatrix, labelMatrix)

		local functionToApply = function (generatedLabelValue, labelValue) return math.abs(generatedLabelValue - labelValue) end

		local absoluteErrorTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelMatrix, labelMatrix)

		local sumAbsoluteErrorValue = AqwamTensorLibrary:sum(absoluteErrorTensor)

		return sumAbsoluteErrorValue

	end,
	
	["BinaryCrossEntropy"] = function(generatedLabelMatrix, labelMatrix)

		local functionToApply = function (generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue) + (1 - labelValue) * math.log(1 - generatedLabelValue)) end

		local binaryCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelMatrix, labelMatrix)

		local sumBinaryCrossEntropyValue = AqwamTensorLibrary:sum(binaryCrossEntropyTensor)

		return sumBinaryCrossEntropyValue

	end,
	
	["CategoricalCrossEntropy"] = function(generatedLabelMatrix, labelMatrix)

		local functionToApply = function (generatedLabelValue, labelValue) return -(labelValue * math.log(generatedLabelValue)) end

		local categoricalCrossEntropyTensor = AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelMatrix, labelMatrix)

		local sumCategoricalCrossEntropyValue = AqwamTensorLibrary:sum(categoricalCrossEntropyTensor)

		return sumCategoricalCrossEntropyValue

	end,
	
}

local elementWiseActivationFunctionList = {

	["Sigmoid"] = function(z) return 1/(1 + math.exp(-1 * z)) end,

	["Tanh"] = function (z) return math.tanh(z) end,

	["ReLU"] = function (z) return math.max(0, z) end,

	["LeakyReLU"] = function (z) return math.max((0.01 * z), z) end,

	["ELU"] = function (z) return if (z > 0) then z else (0.01 * (math.exp(z) - 1)) end,

	["Gaussian"] = function (z) return math.exp(-math.pow(z, 2)) end,

	["SiLU"] = function (z) return z / (1 + math.exp(-z)) end,

	["Mish"] = function (z) return z * math.tanh(math.log(1 + math.exp(z))) end,

	["BinaryStep"] = function (z) return ((z > 0) and 1) or 0 end

}

local activationFunctionList = {

	["Softmax"] = function (zMatrix)

		local exponentZMatrix = AqwamTensorLibrary:applyFunction(math.exp, zMatrix)

		local exponentZSumMatrix = AqwamTensorLibrary:sum(exponentZMatrix, 2)

		local aMatrix = AqwamTensorLibrary:divide(exponentZMatrix, exponentZSumMatrix)

		return aMatrix

	end,

	["StableSoftmax"] = function (zMatrix)

		local normalizedZMatrix = AqwamTensorLibrary:createTensor({#zMatrix, #zMatrix[1]})

		for i = 1, #zMatrix, 1 do

			local zVector = {zMatrix[i]}

			local highestZValue = AqwamTensorLibrary:findMaximumValue(zVector)

			local subtractedZVector = AqwamTensorLibrary:subtract(zVector, highestZValue)

			normalizedZMatrix[i] = subtractedZVector[1]

		end

		local exponentZMatrix = AqwamTensorLibrary:applyFunction(math.exp, normalizedZMatrix)

		local exponentZSumMatrix = AqwamTensorLibrary:sum(exponentZMatrix, 2)

		local aMatrix = AqwamTensorLibrary:divide(exponentZMatrix, exponentZSumMatrix)

		return aMatrix

	end,

	["None"] = function (zMatrix) return zMatrix end,

}

local lossFunctionList = {
	
	["MeanSquaredError"] = function(generatedLabelMatrix, labelMatrix)

		local lossTensor = AqwamTensorLibrary:subtract(generatedLabelMatrix, labelMatrix)

		return AqwamTensorLibrary:multiply(2, lossTensor)

	end,

	["MeanAbsoluteError"] = function(generatedLabelMatrix, labelMatrix)

		return AqwamTensorLibrary:subtract(generatedLabelMatrix, labelMatrix)

	end,

	["BinaryCrossEntropy"] = function(generatedLabelMatrix, labelMatrix)

		local functionToApply = function (generatedLabelValue, labelValue) return ((generatedLabelValue - labelValue) / (generatedLabelValue * (1 - generatedLabelValue))) end

		return AqwamTensorLibrary:applyFunction(functionToApply, generatedLabelMatrix, labelMatrix)

	end,

	["CategoricalCrossEntropy"] = function(generatedLabelMatrix, labelMatrix)

		return AqwamTensorLibrary:subtract(generatedLabelMatrix, labelMatrix)

	end,
	
}

local elementWiseActivationFunctionDerivativeList = {

	["Sigmoid"] = function (a) return (a * (1 - a)) end,

	["Tanh"] = function (a) return (1 - math.pow(a, 2)) end,

	["ReLU"] = function (z) if (z > 0) then return 1 else return 0 end end,

	["LeakyReLU"] = function (z) if (z > 0) then return 1 else return 0.01 end end,

	["ELU"] = function (z) if (z > 0) then return 1 else return 0.01 * math.exp(z) end end,

	["Gaussian"] = function (z) return -2 * z * math.exp(-math.pow(z, 2)) end,

	["SiLU"] = function (z) return (1 + math.exp(-z) + (z * math.exp(-z))) / (1 + math.exp(-z))^2 end,

	["Mish"] = function (z) return math.exp(z) * (math.exp(3 * z) + 4 * math.exp(2 * z) + (6 + 4 * z) * math.exp(z) + 4 * (1 + z)) / math.pow((1 + math.pow((math.exp(z) + 1), 2)), 2) end

}

local activationFunctionDerivativeList = {

	["BinaryStep"] = function (aMatrix, zMatrix) return AqwamTensorLibrary:createTensor({#zMatrix, #zMatrix[1]}, 0) end,

	["Softmax"] = function (aMatrix, zMatrix)

		local numberOfRows, numberOfColumns = #aMatrix, #aMatrix[1]

		local derivativeMatrix = AqwamTensorLibrary:createTensor({numberOfRows, numberOfColumns}, 0)

		for i = 1, numberOfRows, 1 do

			for j = 1, numberOfColumns do

				for k = 1, numberOfColumns do

					if (j == k) then

						derivativeMatrix[i][j] += aMatrix[i][j] * (1 - aMatrix[i][k])

					else

						derivativeMatrix[i][j] += -aMatrix[i][j] * aMatrix[i][k]

					end

				end

			end

		end

		return derivativeMatrix

	end,

	["StableSoftmax"] = function (aMatrix, zMatrix)

		local numberOfRows, numberOfColumns = #aMatrix, #aMatrix[1]

		local derivativeMatrix = AqwamTensorLibrary:createTensor({numberOfRows, numberOfColumns}, 0)

		for i = 1, numberOfRows, 1 do

			for j = 1, numberOfColumns do

				for k = 1, numberOfColumns do

					if (j == k) then

						derivativeMatrix[i][j] += aMatrix[i][j] * (1 - aMatrix[i][k])

					else

						derivativeMatrix[i][j] += -aMatrix[i][j] * aMatrix[i][k]

					end

				end

			end

		end

		return derivativeMatrix

	end,

	["None"] = function (aMatrix, zMatrix) return AqwamTensorLibrary:createTensor({#zMatrix, #zMatrix[1]}, 1) end,

}

local cutOffListForScalarValues = {

	["Sigmoid"] = function (a) return (a >= 0.5) end,

	["Tanh"] = function (a) return (a >= 0) end,

	["ReLU"] = function (a) return (a >= 0) end,

	["LeakyReLU"] = function (a) return (a >= 0) end,

	["ELU"] = function (a) return (a >= 0) end,

	["Gaussian"] = function (a) return (a >= 0.5) end,

	["SiLU"] = function (a) return (a >= 0) end,

	["Mish"] = function (a) return (a >= 0) end,

	["BinaryStep"] = function (a) return (a > 0) end,

	["Softmax"] = function (a) return (a >= 0.5) end,

	["StableSoftmax"] = function (a) return (a >= 0.5) end,

	["None"] = function (a) return (a >= 0) end,

}

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

function NeuralNetworkModel:getActivationLayerAtFinalLayer()
	
	local activationFunctionArray = self.activationFunctionArray

	local finalLayerActivationFunctionName
	
	for layerNumber = #activationFunctionArray, 1, -1 do

		finalLayerActivationFunctionName = activationFunctionArray[layerNumber]

		if (finalLayerActivationFunctionName ~= "None") then break end

	end

	return finalLayerActivationFunctionName

end

function NeuralNetworkModel:convertLabelVectorToLogisticMatrix(labelVector)
	
	local ModelParameters = self.ModelParameters

	local ClassesList = self.ClassesList
	
	local numberOfNeuronsAtFinalLayer = #ModelParameters[#ModelParameters][1]

	if (numberOfNeuronsAtFinalLayer ~= #ClassesList) then error("The number of classes are not equal to number of neurons. Please adjust your last layer using setLayers() function.") end

	if (typeof(labelVector) == "number") then

		labelVector = {{labelVector}}

	end

	local incorrectLabelValue

	local activationFunctionAtFinalLayer = self:getActivationLayerAtFinalLayer()

	if (activationFunctionAtFinalLayer == "Tanh") or (activationFunctionAtFinalLayer == "ELU") then

		incorrectLabelValue = -1

	else

		incorrectLabelValue = 0

	end

	local numberOfData = #labelVector

	local logisticMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfNeuronsAtFinalLayer}, incorrectLabelValue)

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

local function dropoutInputMatrix(inputMatrix, hasBiasNeuron, dropoutRate, doNotDropoutNeurons) -- Don't bother using the applyFunction from AqwamMatrixLibrary. Otherwise, you cannot apply dropout at the same index for both z matrix and activation matrix.

	if (doNotDropoutNeurons) or (dropoutRate == 0) then return inputMatrix end

	local numberOfData = #inputMatrix

	local numberOfFeatures = #inputMatrix[1]

	local nonDropoutRate = 1 - dropoutRate

	local scaleFactor = (1 / nonDropoutRate)

	for data = 1, numberOfData, 1 do

		for neuron = (hasBiasNeuron + 1), numberOfFeatures, 1 do -- Dropout are not applied to bias, so we skip them.

			if (math.random() > nonDropoutRate) then

				inputMatrix[data][neuron] = 0

			else

				inputMatrix[data][neuron] = inputMatrix[data][neuron] * scaleFactor

			end

		end

	end

	return inputMatrix

end

function NeuralNetworkModel:forwardPropagate(featureMatrix, saveAllArrays, doNotDropoutNeurons)

	if (self.ModelParameters == nil) then self:generateLayers() end

	local ModelParameters = self.ModelParameters

	local numberOfLayers = #self.numberOfNeuronsArray

	local hasBiasNeuronArray = self.hasBiasNeuronArray

	local activationFunctionArray = self.activationFunctionArray

	local dropoutRateArray = self.dropoutRateArray

	local forwardPropagateArray = {}

	local zArray = {}

	local activationFunctionName = activationFunctionArray[1]

	local elementWiseActivationFunction = elementWiseActivationFunctionList[activationFunctionName]

	local layerZMatrix = featureMatrix

	local inputMatrix = featureMatrix

	local numberOfData = #featureMatrix

	if (elementWiseActivationFunction) then

		inputMatrix = AqwamTensorLibrary:applyFunction(elementWiseActivationFunction, layerZMatrix)

	else

		inputMatrix = activationFunctionList[activationFunctionName](layerZMatrix)

	end

	inputMatrix = dropoutInputMatrix(inputMatrix, hasBiasNeuronArray[1], dropoutRateArray[1], doNotDropoutNeurons)

	table.insert(zArray, inputMatrix)

	table.insert(forwardPropagateArray, inputMatrix) -- don't remove this! otherwise the code won't work!

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local weightMatrix = ModelParameters[layerNumber]

		local hasBiasNeuron = hasBiasNeuronArray[layerNumber + 1]

		local activationFunctionName = activationFunctionArray[layerNumber + 1]

		local dropoutRate = dropoutRateArray[layerNumber + 1]

		local elementWiseActivationFunction = elementWiseActivationFunctionList[activationFunctionName]

		layerZMatrix = AqwamTensorLibrary:dotProduct(inputMatrix, weightMatrix)

		if (typeof(layerZMatrix) == "number") then layerZMatrix = {{layerZMatrix}} end

		if (elementWiseActivationFunction) then

			inputMatrix = AqwamTensorLibrary:applyFunction(elementWiseActivationFunction, layerZMatrix)

		else

			inputMatrix = activationFunctionList[activationFunctionName](layerZMatrix)

		end

		if (hasBiasNeuron == 1) then

			for data = 1, numberOfData, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron.

		end

		inputMatrix = dropoutInputMatrix(inputMatrix, hasBiasNeuron, dropoutRate, doNotDropoutNeurons)

		table.insert(zArray, layerZMatrix)

		table.insert(forwardPropagateArray, inputMatrix)

		self:sequenceWait()

	end

	if (saveAllArrays) then

		self.forwardPropagateArray = forwardPropagateArray

		self.zArray = zArray

	end

	return inputMatrix, forwardPropagateArray, zArray

end

function NeuralNetworkModel:backwardPropagate(lossMatrix)

	local forwardPropagateArray = self.forwardPropagateArray

	local zArray = self.zArray

	if (forwardPropagateArray == nil) then error("Array not found for forward propagation.") end

	if (zArray == nil) then error("Array not found for z matrix.") end

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local costFunctionDerivativeMatrixArray = {}

	local errorMatrixArray = {}

	local numberOfData = #lossMatrix

	local ModelParameters = self.ModelParameters

	local numberOfLayers = #self.numberOfNeuronsArray

	local activationFunctionArray = self.activationFunctionArray

	local hasBiasNeuronArray = self.hasBiasNeuronArray

	local activationFunctionName = activationFunctionArray[numberOfLayers]

	local elementWiseActivationFunctionDerivative = elementWiseActivationFunctionDerivativeList[activationFunctionName]

	local lastActivationMatrix = forwardPropagateArray[numberOfLayers]

	local lastZMatrix = zArray[numberOfLayers]

	local derivativeMatrix

	if (elementWiseActivationFunctionDerivative) then

		derivativeMatrix = AqwamTensorLibrary:applyFunction(elementWiseActivationFunctionDerivative, lastZMatrix)

	else

		derivativeMatrix = activationFunctionDerivativeList[activationFunctionName](lastActivationMatrix, lastZMatrix)

	end

	local layerCostMatrix = AqwamTensorLibrary:multiply(lossMatrix, derivativeMatrix)

	table.insert(errorMatrixArray, layerCostMatrix)

	for layerNumber = (numberOfLayers - 1), 2, -1 do

		activationFunctionName = activationFunctionArray[layerNumber]

		local hasBiasNeuronOnNextLayer = hasBiasNeuronArray[layerNumber + 1]

		local layerMatrix = AqwamTensorLibrary:transpose(ModelParameters[layerNumber])

		local partialErrorMatrix = AqwamTensorLibrary:dotProduct(layerCostMatrix, layerMatrix)

		local elementWiseActivationFunctionDerivative = elementWiseActivationFunctionDerivativeList[activationFunctionName]

		local currentActivationMatrix = forwardPropagateArray[layerNumber]

		local currentZMatrix = zArray[layerNumber]

		local derivativeMatrix

		if (elementWiseActivationFunctionDerivative) then

			derivativeMatrix = AqwamTensorLibrary:applyFunction(elementWiseActivationFunctionDerivative, currentZMatrix)

		else

			derivativeMatrix = activationFunctionDerivativeList[activationFunctionName](currentActivationMatrix, currentZMatrix)

		end

		if (hasBiasNeuronOnNextLayer == 1) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

			for i = 1, #derivativeMatrix, 1 do derivativeMatrix[i][1] = 0 end

		end

		layerCostMatrix = AqwamTensorLibrary:multiply(partialErrorMatrix, derivativeMatrix)

		table.insert(errorMatrixArray, 1, layerCostMatrix)

		self:sequenceWait()

	end

	for layer = 1, (numberOfLayers - 1), 1 do

		local activationLayerMatrix = AqwamTensorLibrary:transpose(forwardPropagateArray[layer])

		local errorMatrix = errorMatrixArray[layer]

		local costFunctionDerivatives = AqwamTensorLibrary:dotProduct(activationLayerMatrix, errorMatrix)

		if (type(costFunctionDerivatives) == "number") then costFunctionDerivatives = {{costFunctionDerivatives}} end

		table.insert(costFunctionDerivativeMatrixArray, costFunctionDerivatives)

		self:sequenceWait()

	end

	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrixArray end

	return costFunctionDerivativeMatrixArray

end

function NeuralNetworkModel:gradientDescent(costFunctionDerivativeMatrixArray, numberOfData)

	local NewModelParameters = {}

	local numberOfLayers = #self.numberOfNeuronsArray

	local learningRateArray = self.learningRateArray

	local OptimizerArray = self.OptimizerArray

	local RegularizerArray = self.RegularizerArray

	local hasBiasNeuronArray = self.hasBiasNeuronArray

	local ModelParameters = self.ModelParameters

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local learningRate = learningRateArray[layerNumber + 1]

		local Regularizer = RegularizerArray[layerNumber + 1]

		local Optimizer = OptimizerArray[layerNumber + 1]

		local costFunctionDerivativeMatrix = costFunctionDerivativeMatrixArray[layerNumber]

		local hasBiasNeuronOnNextLayer = hasBiasNeuronArray[layerNumber + 1]

		if (type(costFunctionDerivativeMatrix) == "number") then costFunctionDerivativeMatrix = {{costFunctionDerivativeMatrix}} end

		local weightMatrix = ModelParameters[layerNumber]

		if (Regularizer ~= 0) then

			local regularizationDerivativeMatrix = Regularizer:calculate(weightMatrix)

			costFunctionDerivativeMatrix = AqwamTensorLibrary:add(costFunctionDerivativeMatrix, regularizationDerivativeMatrix)

		end

		costFunctionDerivativeMatrix = AqwamTensorLibrary:divide(costFunctionDerivativeMatrix, numberOfData)

		if (Optimizer ~= 0) then

			costFunctionDerivativeMatrix = Optimizer:calculate(learningRate, costFunctionDerivativeMatrix)

		else

			costFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, costFunctionDerivativeMatrix)

		end

		local newWeightMatrix = AqwamTensorLibrary:subtract(weightMatrix, costFunctionDerivativeMatrix)

		if (hasBiasNeuronOnNextLayer == 1) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

			for i = 1, #newWeightMatrix, 1 do newWeightMatrix[i][1] = 0 end

		end

		table.insert(NewModelParameters, newWeightMatrix)

	end
	
	self.ModelParameters = NewModelParameters

end

function NeuralNetworkModel:update(lossMatrix, clearAllArrays)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrixArray = self:backwardPropagate(lossMatrix)

	self:gradientDescent(costFunctionDerivativeMatrixArray, numberOfData)

	if (clearAllArrays) then

		self.forwardPropagateArray = nil

		self.zArray = nil

	end

end

function NeuralNetworkModel:calculateCost(allOutputsMatrix, logisticMatrix)
	
	local numberOfLayers = #self.numberOfNeuronsArray

	local RegularizerArray = self.RegularizerArray

	local ModelParameters = self.ModelParameters
	
	local CostFunctionToApply = costFunctionList[self.costFunction]

	local totalCost = CostFunctionToApply(allOutputsMatrix, logisticMatrix)

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local Regularizer = RegularizerArray[layerNumber + 1]

		if (Regularizer ~=  0) then totalCost = totalCost + Regularizer:calculateCost(ModelParameters[layerNumber]) end

	end

	local cost = totalCost / #logisticMatrix

	return cost

end

function NeuralNetworkModel:fetchValueFromScalar(outputVector)

	local value = outputVector[1][1]

	local activationFunctionAtFinalLayer = self:getActivationLayerAtFinalLayer()

	local isValueOverCutOff = cutOffListForScalarValues[activationFunctionAtFinalLayer](value)

	local classIndex = (isValueOverCutOff and 2) or 1

	local predictedLabel = self.ClassesList[classIndex]

	return predictedLabel, value

end

function NeuralNetworkModel:fetchHighestValueInVector(outputVector)

	local dimensionIndexArray, highestValue = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(outputVector)

	if (not dimensionIndexArray) then return nil, highestValue end

	local predictedLabel = self.ClassesList[dimensionIndexArray[2]]

	return predictedLabel, highestValue

end

function NeuralNetworkModel:getLabelFromOutputMatrix(outputMatrix)

	local numberOfData = #outputMatrix
	
	local numberOfNeuronsArray = self.numberOfNeuronsArray

	local numberOfNeuronsAtFinalLayer = numberOfNeuronsArray[#numberOfNeuronsArray]

	local predictedLabelVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestValueVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local highestValue

	local outputVector

	local classIndex

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		if (numberOfNeuronsAtFinalLayer == 1) then

			predictedLabel, highestValue = self:fetchValueFromScalar(outputVector)

		else

			predictedLabel, highestValue = self:fetchHighestValueInVector(outputVector)

		end

		predictedLabelVector[i][1] = predictedLabel

		highestValueVector[i][1] = highestValue

	end

	return predictedLabelVector, highestValueVector

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, classesList)

	for i = 1, #labelVector, 1 do

		if table.find(classesList, labelVector[i][1]) then continue end

		return true

	end

	return false

end

function NeuralNetworkModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewNeuralNetworkModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewNeuralNetworkModel, NeuralNetworkModel)

	NewNeuralNetworkModel:setName("NeuralNetwork")
	
	NewNeuralNetworkModel.costFunction = parameterDictionary.costFunction or defaultCostFunction

	NewNeuralNetworkModel.ClassesList = parameterDictionary.ClassesList or {}

	NewNeuralNetworkModel.numberOfNeuronsArray = parameterDictionary.numberOfNeuronsArray or {}

	NewNeuralNetworkModel.RegularizerArray = parameterDictionary.RegularizerArray or {}

	NewNeuralNetworkModel.OptimizerArray = parameterDictionary.OptimizerArray or {}

	NewNeuralNetworkModel.hasBiasNeuronArray = parameterDictionary.hasBiasNeuronArray or {}

	NewNeuralNetworkModel.learningRateArray = parameterDictionary.learningRateArray or {}

	NewNeuralNetworkModel.activationFunctionArray = parameterDictionary.activationFunctionArray or {}

	NewNeuralNetworkModel.dropoutRateArray = parameterDictionary.dropoutRateArray or {}

	return NewNeuralNetworkModel

end

function NeuralNetworkModel:generateLayers()
	
	local numberOfNeuronsArray = self.numberOfNeuronsArray

	local numberOfLayers = #numberOfNeuronsArray

	if (numberOfLayers == 1) then error("There is only one layer!") end

	local ModelParameters = {}
	
	local hasBiasNeuronArray = self.hasBiasNeuronArray

	for layer = 1, (numberOfLayers - 1), 1 do

		local numberOfCurrentLayerNeurons = numberOfNeuronsArray[layer]

		if (hasBiasNeuronArray[layer] == 1) then numberOfCurrentLayerNeurons += 1 end -- 1 is added for bias

		local numberOfNextLayerNeurons = numberOfNeuronsArray[layer + 1]

		local hasBiasNeuronOnNextLayer = hasBiasNeuronArray[layer + 1] 

		if (hasBiasNeuronOnNextLayer == 1) then numberOfNextLayerNeurons += 1 end

		local weightMatrix = self:initializeMatrixBasedOnMode({numberOfCurrentLayerNeurons, numberOfNextLayerNeurons}, {0, hasBiasNeuronOnNextLayer}) -- Since no outputs are going into the bias neuron, it should not be considered as an input neuron. So the bias column needed to be excluded for our weight initialization.

		table.insert(ModelParameters, weightMatrix)

	end

	self.ModelParameters = ModelParameters

end

function NeuralNetworkModel:createLayers(numberOfNeuronsArray, activationFunction, learningRate, OptimizerArray, RegularizerArray, dropoutRate)

	local learningRateType = typeof(learningRate)

	local activationFunctionType = typeof(activationFunction)

	local dropoutRateType = typeof(dropoutRate)

	OptimizerArray = OptimizerArray or {}

	RegularizerArray = RegularizerArray or {}

	if (activationFunctionType ~= "nil") and (activationFunctionType ~= "string") then error("Invalid input for activation function!") end

	if (learningRateType ~= "nil") and (learningRateType ~= "number") then error("Invalid input for learning rate!") end

	if (dropoutRateType ~= "nil") and (dropoutRateType ~= "number") then error("Invalid input for dropout rate!") end

	activationFunction = activationFunction or defaultActivationFunction

	learningRate = learningRate or defaultLearningRate

	dropoutRate = dropoutRate or defaultDropoutRate

	self.ModelParameters = nil

	local numberOfNeuronsArray = numberOfNeuronsArray

	local hasBiasNeuronArray = {}

	local learningRateArray = {}

	local activationFunctionArray = {}
	
	local OptimizerArray = {}

	local RegularizerArray = {}

	local dropoutRateArray = {}

	local numberOfLayers = #numberOfNeuronsArray

	for layer = 1, numberOfLayers, 1 do
		
		hasBiasNeuronArray[layer] = ((layer == numberOfLayers) and 0) or 1
		
		learningRateArray[layer] = ((layer == 1) and 0) or learningRate

		activationFunctionArray[layer] = ((layer == 1) and "None") or activationFunction

		OptimizerArray[layer] = OptimizerArray[layer] or 0

		RegularizerArray[layer] = RegularizerArray[layer] or 0
		
		dropoutRateArray[layer] = dropoutRate

	end
	
	self.hasBiasNeuronArray = hasBiasNeuronArray

	self.learningRateArray = learningRateArray

	self.activationFunctionArray = activationFunctionArray

	self.OptimizerArray = OptimizerArray

	self.RegularizerArray = RegularizerArray
	
	self.dropoutRateArray = dropoutRateArray

	self:generateLayers()

end

function NeuralNetworkModel:addLayer(numberOfNeurons, hasBiasNeuron, activationFunction, learningRate, Optimizer, Regularizer, dropoutRate)

	local numberOfNeuronsArray = self.numberOfNeuronsArray

	local isFirstLayer = (#numberOfNeuronsArray == 0)

	if (isFirstLayer) and (not activationFunction) then activationFunction = "None" end

	if (isFirstLayer) and (not learningRate) then learningRate = 0 end

	layerPropertyValueTypeCheckingFunctionList["NumberOfNeurons"](numberOfNeurons)

	layerPropertyValueTypeCheckingFunctionList["HasBias"](hasBiasNeuron)

	layerPropertyValueTypeCheckingFunctionList["ActivationFunction"](activationFunction)

	layerPropertyValueTypeCheckingFunctionList["LearningRate"](learningRate)

	layerPropertyValueTypeCheckingFunctionList["DropoutRate"](dropoutRate)

	hasBiasNeuron = self:getValueOrDefaultValue(hasBiasNeuron, true)

	hasBiasNeuron = (hasBiasNeuron and 1) or 0

	learningRate = learningRate or defaultLearningRate

	activationFunction = activationFunction or defaultActivationFunction

	dropoutRate = dropoutRate or defaultDropoutRate

	table.insert(numberOfNeuronsArray, numberOfNeurons)

	table.insert(self.hasBiasNeuronArray, hasBiasNeuron)

	table.insert(self.activationFunctionArray, activationFunction)

	table.insert(self.learningRateArray, learningRate)

	table.insert(self.OptimizerArray, Optimizer or 0)

	table.insert(self.RegularizerArray, Regularizer or 0)

	table.insert(self.dropoutRateArray, dropoutRate)

end

function NeuralNetworkModel:setLayer(layerNumber, hasBiasNeuron, activationFunction, learningRate, Optimizer, Regularizer, dropoutRate)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsArray)  then

		error("The layer number exceeds the number of layers!") 

	end 

	layerPropertyValueTypeCheckingFunctionList["HasBias"](hasBiasNeuron)

	layerPropertyValueTypeCheckingFunctionList["ActivationFunction"](activationFunction)

	layerPropertyValueTypeCheckingFunctionList["LearningRate"](learningRate)

	layerPropertyValueTypeCheckingFunctionList["DropoutRate"](dropoutRate)

	hasBiasNeuron = self:getValueOrDefaultValue(hasBiasNeuron,  self.hasBiasNeuronArray[layerNumber])

	hasBiasNeuron = (hasBiasNeuron and 1) or 0

	Regularizer = self:getValueOrDefaultValue(Regularizer,  self.RegularizerArray[layerNumber])

	Regularizer = Regularizer or 0

	Optimizer = self:getValueOrDefaultValue(Optimizer,  self.OptimizerArray[layerNumber])

	Optimizer = Optimizer or 0

	self.hasBiasNeuronArray[layerNumber] = hasBiasNeuron

	self.activationFunctionArray[layerNumber] = activationFunction or self.activationFunctionArray[layerNumber] 

	self.learningRateArray[layerNumber] = activationFunction or self.learningRateArray[layerNumber] 

	self.OptimizerArray[layerNumber] = Optimizer

	self.RegularizerArray[layerNumber] = Regularizer

	self.dropoutRateArray[layerNumber] = dropoutRate or self.dropoutRateArray[layerNumber]

end

function NeuralNetworkModel:setLayerProperty(layerNumber, property, value)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsArray)  then

		error("The layer number exceeds the number of layers!") 

	end 

	if (property == "HasBias") then

		layerPropertyValueTypeCheckingFunctionList["HasBias"](value)

		local hasBiasNeuron = self:getValueOrDefaultValue(value,  self.hasBiasNeuronArray[layerNumber])

		hasBiasNeuron = (hasBiasNeuron and 1) or 0

		self.hasBiasNeuronArray[layerNumber] = hasBiasNeuron

	elseif (property == "ActivationFunction") then

		layerPropertyValueTypeCheckingFunctionList["ActivationFunction"](value)

		self.activationFunctionArray[layerNumber] = value or self.activationFunctionArray[layerNumber]

	elseif (property == "LearningRate") then

		layerPropertyValueTypeCheckingFunctionList["LearningRate"](value)

		self.learningRateArray[layerNumber] = value or self.learningRateArray[layerNumber]

	elseif (property == "Optimizer") then

		value = self:getValueOrDefaultValue(value, self.OptimizerArray[layerNumber])

		value = value or 0

		self.OptimizerArray[layerNumber] = value

	elseif (property == "Regularizer") then

		value = self:getValueOrDefaultValue(value, self.OptimizerArray[layerNumber])

		value = value or 0

		self.RegularizerArray[layerNumber] = value or 0

	elseif (property == "DropoutRate") then

		layerPropertyValueTypeCheckingFunctionList["DropoutRate"](value)

		self.dropoutRateArray[layerNumber] = value or self.dropoutRateArray[layerNumber]

	else

		warn("Layer property does not exists. Did not change the layer's properties.")

	end

end

function NeuralNetworkModel:getLayerProperty(layerNumber, property)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsArray)  then

		error("The layer number exceeds the number of layers!") 

	end 

	if (property == "HasBias") then

		return (self.hasBiasNeuronArray[layerNumber] == 1)

	elseif (property == "ActivationFunction") then

		return self.activationFunctionArray[layerNumber]

	elseif (property == "LearningRate") then

		return self.learningRateArray[layerNumber]

	elseif (property == "Optimizer") then

		local Optimizer = self.OptimizerArray[layerNumber]

		if (Optimizer ~= 0) then

			return Optimizer

		else

			return nil

		end

	elseif (property == "Regularizer") then

		local Regularizer = self.RegularizerArray[layerNumber]

		if (Regularizer ~= 0) then

			return Regularizer

		else

			return nil

		end

	elseif (property == "DropoutRate") then

		return self.dropoutRateArray[layerNumber]

	else

		warn("Layer property does not exists. Returning nil value.")

		return nil

	end

end

function NeuralNetworkModel:getLayer(layerNumber)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsArray) then

		error("The layer number exceeds the number of layers!") 

	end 

	local Optimizer = self.OptimizerArray[layerNumber]

	if (Optimizer == 0) then

		Optimizer = nil

	end

	local Regularizer = self.RegularizerArray[layerNumber]

	if (Regularizer == 0) then

		Regularizer = nil

	end

	return self.numberOfNeuronsArray[layerNumber], (self.hasBiasNeuronArray[layerNumber] == 1), self.activationFunctionArray[layerNumber], self.learningRateArray[layerNumber], Optimizer, Regularizer, self.dropoutRateArray[layerNumber]

end

function NeuralNetworkModel:getTotalNumberOfNeurons(layerNumber)

	return self.numberOfNeuronsArray[layerNumber] + self.hasBiasNeuronArray[layerNumber]

end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

function NeuralNetworkModel:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(self.ClassesList)

		if (areNumbersOnly) then table.sort(self.ClassesList, function(a,b) return a < b end) end

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector.") end

	end

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)

	return logisticMatrix

end

local function mergeLayers(numberOfNeurons, initialNeuronIndex, currentWeightMatrixLeft, currentWeightMatrixRight, currentWeightMatrixToAdd, nextWeightMatrixTop, nextWeightMatrixToAdd, nextWeightMatrixBottom)

	local newCurrentWeightMatrix
	local newNextWeightMatrix

	if (numberOfNeurons < initialNeuronIndex) then

		newCurrentWeightMatrix = AqwamTensorLibrary:columnConcatenate(currentWeightMatrixLeft, currentWeightMatrixToAdd, currentWeightMatrixRight)
		newNextWeightMatrix = AqwamTensorLibrary:rowConcatenate(nextWeightMatrixTop, nextWeightMatrixToAdd, nextWeightMatrixBottom)

	else

		newCurrentWeightMatrix = AqwamTensorLibrary:columnConcatenate(currentWeightMatrixLeft, currentWeightMatrixRight, currentWeightMatrixToAdd)
		newNextWeightMatrix = AqwamTensorLibrary:rowConcatenate(nextWeightMatrixTop, nextWeightMatrixBottom, nextWeightMatrixToAdd)

	end

	return newCurrentWeightMatrix, newNextWeightMatrix

end

function NeuralNetworkModel:evolveLayerSize(layerNumber, initialNeuronIndex, size)
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters == nil) then error("No Model Parameters!") end

	if (#ModelParameters == 0) then 

		self.ModelParameters = nil
		error("No Model Parameters!") 

	end

	local numberOfLayers = #self.numberOfNeuronsArray -- DON'T FORGET THAT IT DOES NOT INCLUDE BIAS!

	if (layerNumber > numberOfLayers) then error("Layer number exceeds this model's number of layers.") end

	local hasBiasNeuronValue = self.hasBiasNeuronArray[layerNumber]

	local numberOfNeurons = self.numberOfNeuronsArray[layerNumber] + hasBiasNeuronValue

	local currentWeightMatrix
	local nextWeightMatrix

	if (layerNumber == numberOfLayers) then

		currentWeightMatrix = ModelParameters[numberOfLayers - 1]

	elseif (layerNumber > 1) and (layerNumber < numberOfLayers) then

		currentWeightMatrix = ModelParameters[layerNumber - 1]
		nextWeightMatrix = ModelParameters[layerNumber]

	else

		currentWeightMatrix = ModelParameters[1]
		nextWeightMatrix = ModelParameters[2]

	end

	initialNeuronIndex = initialNeuronIndex or numberOfNeurons

	if (initialNeuronIndex > numberOfNeurons) then error("The index exceeds this layer's number of neurons.") end

	local hasNextLayer = (typeof(nextWeightMatrix) ~= "nil")

	local absoluteSize = math.abs(size)

	local secondNeuronIndex = initialNeuronIndex + size + 1
	local thirdNeuronIndex = initialNeuronIndex + 2

	local newCurrentWeightMatrix
	local newNextWeightMatrix

	local currentWeightMatrixLeft
	local currentWeightMatrixRight

	local nextWeightMatrixTop
	local nextWeightMatrixBottom

	local currentWeightMatrixToAdd
	local nextWeightMatrixToAdd

	if (size == 0) then

		error("Size is zero!")

	elseif (size < 0) and (numberOfNeurons == 0)  then

		error("No neurons to remove!")

	elseif (size < 0) and ((initialNeuronIndex + size) < 0) then

		error("Size is too large!")

	elseif (initialNeuronIndex == 0) and (size > 0) and (hasNextLayer) then

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode({#currentWeightMatrix, size})
		nextWeightMatrixToAdd =  self:initializeMatrixBasedOnMode({size, #nextWeightMatrix[1]})

		newCurrentWeightMatrix = AqwamTensorLibrary:concatenate(currentWeightMatrix, currentWeightMatrixToAdd, 2)
		newNextWeightMatrix = AqwamTensorLibrary:concatenate(nextWeightMatrix, nextWeightMatrixToAdd, 1)

	elseif (initialNeuronIndex == 0) and (size > 0) and (not hasNextLayer) then

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode(#currentWeightMatrix, size)
		newCurrentWeightMatrix = AqwamTensorLibrary:concatenate(currentWeightMatrixToAdd, currentWeightMatrix, 2)

	elseif (initialNeuronIndex > 0) and (size > 0) and (hasNextLayer) then

		currentWeightMatrixLeft = AqwamTensorLibrary:extractColumns(currentWeightMatrix, 1, initialNeuronIndex)
		currentWeightMatrixRight = AqwamTensorLibrary:extractColumns(currentWeightMatrix, initialNeuronIndex + 1, #currentWeightMatrix[1])

		nextWeightMatrixTop = AqwamTensorLibrary:extractRows(nextWeightMatrix, 1, initialNeuronIndex)
		nextWeightMatrixBottom = AqwamTensorLibrary:extractRows(nextWeightMatrix, initialNeuronIndex + 1, #nextWeightMatrix)

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode({#currentWeightMatrix, size})
		nextWeightMatrixToAdd =  self:initializeMatrixBasedOnMode({size, #nextWeightMatrix[1]})

		newCurrentWeightMatrix, newNextWeightMatrix = mergeLayers(numberOfNeurons, initialNeuronIndex, currentWeightMatrixLeft, currentWeightMatrixRight, currentWeightMatrixToAdd, nextWeightMatrixTop, nextWeightMatrixToAdd, nextWeightMatrixBottom)

	elseif (initialNeuronIndex > 0) and (size > 0) and (not hasNextLayer) then

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode(#currentWeightMatrix, size)
		newCurrentWeightMatrix = AqwamTensorLibrary:concatenate(currentWeightMatrix, currentWeightMatrixToAdd, 2)

	elseif (size == -1) and (hasNextLayer) and (numberOfNeurons == 1) then

		newCurrentWeightMatrix = AqwamTensorLibrary:extractColumns(currentWeightMatrix, initialNeuronIndex, initialNeuronIndex)
		newNextWeightMatrix = AqwamTensorLibrary:extractRows(nextWeightMatrix, initialNeuronIndex, initialNeuronIndex)

	elseif (size == -1) and (not hasNextLayer) and (numberOfNeurons == 1) then

		newCurrentWeightMatrix = AqwamTensorLibrary:extractColumns(currentWeightMatrix, initialNeuronIndex, initialNeuronIndex)

	elseif (size < 0) and (hasNextLayer) and (numberOfNeurons >= absoluteSize) then

		currentWeightMatrixLeft = AqwamTensorLibrary:extractColumns(currentWeightMatrix, 1, secondNeuronIndex)
		currentWeightMatrixRight = AqwamTensorLibrary:extractColumns(currentWeightMatrix, thirdNeuronIndex, #currentWeightMatrix[1])

		nextWeightMatrixTop = AqwamTensorLibrary:extractRows(nextWeightMatrix, 1, secondNeuronIndex)
		nextWeightMatrixBottom = AqwamTensorLibrary:extractRows(nextWeightMatrix, thirdNeuronIndex, #nextWeightMatrix)

		newCurrentWeightMatrix = AqwamTensorLibrary:horizontalConcatenate(currentWeightMatrixLeft, currentWeightMatrixRight)
		newNextWeightMatrix = AqwamTensorLibrary:verticalConcatenate(nextWeightMatrixTop, nextWeightMatrixBottom)

	elseif (size < 0) and (not hasNextLayer) and (numberOfNeurons >= absoluteSize) then

		currentWeightMatrixLeft = AqwamTensorLibrary:extractColumns(currentWeightMatrix, 1, secondNeuronIndex)
		currentWeightMatrixRight = AqwamTensorLibrary:extractColumns(currentWeightMatrix, thirdNeuronIndex, #currentWeightMatrix[1])

		newCurrentWeightMatrix = AqwamTensorLibrary:horizontalConcatenate(currentWeightMatrixLeft, currentWeightMatrixRight)

	end

	if (layerNumber == numberOfLayers) then

		ModelParameters[numberOfLayers - 1] = newCurrentWeightMatrix

	elseif (layerNumber > 1) and (layerNumber < numberOfLayers) then

		ModelParameters[layerNumber - 1] = newCurrentWeightMatrix
		ModelParameters[layerNumber] = newNextWeightMatrix

	else

		ModelParameters[1] = newCurrentWeightMatrix
		ModelParameters[2] = newNextWeightMatrix

	end

	self.numberOfNeuronsArray[layerNumber] += size

end

function NeuralNetworkModel:train(featureMatrix, labelVector)

	local numberOfFeatures = #featureMatrix[1]
	
	local numberOfNeuronsArray = self.numberOfNeuronsArray

	local numberOfNeuronsAtInputLayer = numberOfNeuronsArray[1] + numberOfNeuronsArray[1]

	if (numberOfNeuronsAtInputLayer ~= numberOfFeatures) then error("Input layer has " .. numberOfNeuronsAtInputLayer .. " neuron(s), but feature matrix has " .. #featureMatrix[1] .. " features!") end

	if (#featureMatrix ~= #labelVector) then error("Number of rows of feature matrix and the label vector is not the same!") end
	
	local numberOfNeuronsAtFinalLayer = numberOfNeuronsArray[#numberOfNeuronsArray]
	
	local LossFunctionToApply = lossFunctionList[self.costFunction]

	local numberOfIterations = 0

	local cost

	local costArray = {}

	local deltaArray

	local RegularizerDerivatives

	local logisticMatrix

	local activatedOutputsMatrix

	if (not self.ModelParameters) then self:generateLayers() end

	if (#labelVector[1] == 1) and (numberOfNeuronsAtFinalLayer ~= 1) then

		logisticMatrix = self:processLabelVector(labelVector)

	else

		if (#labelVector[1] ~= numberOfNeuronsAtFinalLayer) then error("The number of columns for the label matrix is not equal to number of neurons at final layer!") end

		logisticMatrix = labelVector

	end

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		activatedOutputsMatrix = self:forwardPropagate(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(activatedOutputsMatrix, logisticMatrix)

		end)

		if cost then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossMatrix = LossFunctionToApply(activatedOutputsMatrix, logisticMatrix)

		self:update(lossMatrix, true)

	until (numberOfIterations == self.maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	if (self.autoResetOptimizers) then

		for i, Optimizer in ipairs(self.OptimizerArray) do

			if (Optimizer ~= 0) then Optimizer:reset() end

		end

	end

	return costArray

end

function NeuralNetworkModel:reset()

	for i, Optimizer in ipairs(self.OptimizerArray) do

		if (Optimizer ~= 0) then Optimizer:reset() end

	end

end

function NeuralNetworkModel:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then self:generateLayers() end

	local outputMatrix = self:forwardPropagate(featureMatrix, false, true)

	if (returnOriginalOutput == true) then return outputMatrix end

	local predictedLabelVector, highestValueVector = self:getLabelFromOutputMatrix(outputMatrix)

	return predictedLabelVector, highestValueVector

end

function NeuralNetworkModel:getClassesList()

	return self.ClassesList

end

function NeuralNetworkModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function NeuralNetworkModel:showDetails()
	
	local numberOfNeuronsArray = self.numberOfNeuronsArray
	
	local hasBiasNeuronArray = self.hasBiasNeuronArray
	
	local activationFunctionArray = self.activationFunctionArray
	
	local learningRateArray = self.learningRateArray
	
	local OptimizerArray = self.OptimizerArray
	
	local RegularizerArray = self.RegularizerArray
	
	local dropoutRateArray = self.dropoutRateArray
	
	-- Calculate the maximum length for each column
	local maxLayerLength = string.len("Layer")
	local maxNeuronsLength = string.len("Number Of Neurons")
	local maxBiasLength = string.len("Has Bias Neuron")
	local maxActivationLength = string.len("Activation Function")
	local maxLearningRateLength = string.len("Learning Rate")
	local maxOptimizerLength = string.len("Optimizer Added")
	local maxRegularizerLength = string.len("Regularizer Added")
	local maxDropoutRateLength = string.len("Dropout Rate")

	local hasBias

	local optimizerName = "None"

	local regularizerName = "None"

	for i = 1, #numberOfNeuronsArray, 1 do

		local Optimizer = OptimizerArray[i]

		local Regularizer = RegularizerArray[i]

		if (type(Optimizer) == "table") then optimizerName = Optimizer:getName() end

		if (type(Regularizer) == "table") then regularizerName = Regularizer:getName() end

		maxLayerLength = math.max(maxLayerLength, string.len(tostring(i)))

		maxNeuronsLength = math.max(maxNeuronsLength, string.len(tostring(numberOfNeuronsArray[i])))

		hasBias = (hasBiasNeuronArray[i] == 1)

		maxBiasLength = math.max(maxBiasLength, string.len(tostring(hasBias)))

		maxActivationLength = math.max(maxActivationLength, string.len(activationFunctionArray[i]))

		maxLearningRateLength = math.max(maxLearningRateLength, string.len(tostring(learningRateArray[i])))

		maxOptimizerLength = math.max(maxOptimizerLength, string.len(optimizerName))

		maxRegularizerLength = math.max(maxRegularizerLength, string.len(regularizerName))

		maxDropoutRateLength = math.max(maxDropoutRateLength, string.len(tostring(dropoutRateArray[i])))

	end

	-- Print the array header

	local stringToPrint = ""

	stringToPrint ..= "Layer Details: \n\n"

	stringToPrint ..= "|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxLearningRateLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizerLength) .. "-|-" .. 
		string.rep("-", maxDropoutRateLength) .. "-|" .. 
		"\n"

	stringToPrint ..= "| " .. string.format("%-" .. maxLayerLength .. "s", "Layer") .. " | " ..
		string.format("%-" .. maxNeuronsLength .. "s", "Number Of Neurons") .. " | " ..
		string.format("%-" .. maxBiasLength .. "s", "Has Bias Neuron") .. " | " ..
		string.format("%-" .. maxActivationLength .. "s", "Activation Function") .. " | " ..
		string.format("%-" .. maxLearningRateLength .. "s", "Learning Rate") .. " | " ..
		string.format("%-" .. maxOptimizerLength .. "s", "Optimizer Added") .. " | " ..
		string.format("%-" .. maxRegularizerLength .. "s", "Regularizer Added") .. " | " .. 
		string.format("%-" .. maxDropoutRateLength .. "s", "Dropout Rate") .. " |" .. 
		"\n"


	stringToPrint ..= "|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxLearningRateLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizerLength) .. "-|-" .. 
		string.rep("-", maxDropoutRateLength) .. "-|" .. 
		"\n"

	-- Print the layer details
	for i = 1, #numberOfNeuronsArray, 1 do

		local optimizerName = "None"

		local regularizerName = "None"

		local layerText = "| " .. string.format("%-" .. maxLayerLength .. "s", i) .. " "

		local numberOfNeuronsText = "| " .. string.format("%-" .. maxNeuronsLength .. "s", numberOfNeuronsArray[i]) .. " "

		hasBias = (hasBiasNeuronArray[i] == 1)

		local biasText = "| " .. string.format("%-" .. maxBiasLength .. "s", tostring(hasBias)) .. " "

		local activationFunctionText = "| " .. string.format("%-" .. maxActivationLength .. "s", activationFunctionArray[i]) .. " "

		local learningRateText = "| " .. string.format("%-" .. maxLearningRateLength .. "s", learningRateArray[i]) .. " "

		local Optimizer = OptimizerArray[i]

		local Regularizer = RegularizerArray[i]

		if (type(Optimizer) == "table") then optimizerName = Optimizer:getName() end

		if (type(Regularizer) == "table") then regularizerName = Regularizer:getName() end

		local optimizerText = "| " .. string.format("%-" .. maxOptimizerLength .. "s", optimizerName) .. " "

		local regularizerText = "| " .. string.format("%-" .. maxRegularizerLength .. "s",  regularizerName) .. " "

		local dropoutRateText = "| " .. string.format("%-" .. maxDropoutRateLength .. "s", dropoutRateArray[i]) .. " |"

		local stringPart = layerText .. numberOfNeuronsText .. biasText .. activationFunctionText .. learningRateText .. optimizerText .. regularizerText .. dropoutRateText .. "\n"

		stringToPrint ..= stringPart

	end

	stringToPrint ..= "|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxLearningRateLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizerLength) .. "-|-"..
		string.rep("-", maxDropoutRateLength) .. "-|"..
		"\n\n"

	print(stringToPrint)

end

function NeuralNetworkModel:getNumberOfLayers()

	return #self.numberOfNeuronsArray

end

return NeuralNetworkModel
