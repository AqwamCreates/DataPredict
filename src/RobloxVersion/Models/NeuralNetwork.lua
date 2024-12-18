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

NeuralNetworkModel = {}

NeuralNetworkModel.__index = NeuralNetworkModel

setmetatable(NeuralNetworkModel, GradientMethodBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

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

	["Softmax"] = function (zMatrix) -- apparently roblox doesn't really handle very small values such as math.exp(-1000), so I added a more stable computation exp(a) / exp(b) -> exp (a - b)

		local expMatrix = AqwamMatrixLibrary:applyFunction(math.exp, zMatrix)

		local expSum = AqwamMatrixLibrary:horizontalSum(expMatrix)

		local aMatrix = AqwamMatrixLibrary:divide(expMatrix, expSum)

		return aMatrix

	end,

	["StableSoftmax"] = function (zMatrix)

		local normalizedZMatrix = AqwamMatrixLibrary:createMatrix(#zMatrix, #zMatrix[1])

		for i = 1, #zMatrix, 1 do

			local zVector = {zMatrix[i]}

			local highestZValue = AqwamMatrixLibrary:findMaximumValue(zVector)

			local subtractedZVector = AqwamMatrixLibrary:subtract(zVector, highestZValue)

			normalizedZMatrix[i] = subtractedZVector[1]

		end

		local expMatrix = AqwamMatrixLibrary:applyFunction(math.exp, normalizedZMatrix)

		local expSum = AqwamMatrixLibrary:horizontalSum(expMatrix)

		local aMatrix = AqwamMatrixLibrary:divide(expMatrix, expSum)

		return aMatrix

	end,

	["None"] = function (zMatrix) return zMatrix end,

}

local elementWiseActivationFunctionDerivativeList = {
	
	["ReLU"] = function (z) if (z > 0) then return 1 else return 0 end end,
	
	["LeakyReLU"] = function (z) if (z > 0) then return 1 else return 0.01 end end,
	
	["ELU"] = function (z) if (z > 0) then return 1 else return 0.01 * math.exp(z) end end,
	
	["Gaussian"] = function (z) return -2 * z * math.exp(-math.pow(z, 2)) end,
	
	["SiLU"] = function (z) return (1 + math.exp(-z) + (z * math.exp(-z))) / (1 + math.exp(-z))^2 end,
	
	["Mish"] = function (z) return math.exp(z) * (math.exp(3 * z) + 4 * math.exp(2 * z) + (6 + 4 * z) * math.exp(z) + 4 * (1 + z)) / math.pow((1 + math.pow((math.exp(z) + 1), 2)), 2) end
	
}

local activationFunctionDerivativeList = {

	["Sigmoid"] = function (aMatrix, zMatrix) 

		local sigmoidDerivativeFunction = function (a) return (a * (1 - a)) end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(sigmoidDerivativeFunction, aMatrix)

		return derivativeMatrix

	end,

	["Tanh"] = function (aMatrix, zMatrix)

		local tanhDerivativeFunction = function (a) return (1 - math.pow(a, 2)) end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(tanhDerivativeFunction, aMatrix)

		return derivativeMatrix

	end,

	["BinaryStep"] = function (aMatrix, zMatrix) return AqwamMatrixLibrary:createMatrix(#zMatrix, #zMatrix[1], 0) end,

	["Softmax"] = function (aMatrix, zMatrix)

		local numberOfRows, numberOfColumns = #aMatrix, #aMatrix[1]

		local derivativeMatrix = AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns)

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

		local derivativeMatrix = AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns)

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

	["None"] = function (aMatrix, zMatrix) return AqwamMatrixLibrary:createMatrix(#zMatrix, #zMatrix[1], 1) end,

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

	local finalLayerActivationFunctionName

	for layerNumber = #self.activationFunctionTable, 1, -1 do

		finalLayerActivationFunctionName = self.activationFunctionTable[layerNumber]

		if (finalLayerActivationFunctionName ~= "None") then break end

	end

	return finalLayerActivationFunctionName

end

function NeuralNetworkModel:convertLabelVectorToLogisticMatrix(labelVector)

	local numberOfNeuronsAtFinalLayer = #self.ModelParameters[#self.ModelParameters][1]

	if (numberOfNeuronsAtFinalLayer ~= #self.ClassesList) then error("The number of classes are not equal to number of neurons. Please adjust your last layer using setLayers() function.") end

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

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(#labelVector, numberOfNeuronsAtFinalLayer, incorrectLabelValue)

	local label

	local labelPosition

	for row = 1, #labelVector, 1 do

		label = labelVector[row][1]

		labelPosition = table.find(self.ClassesList, label)

		logisticMatrix[row][labelPosition] = 1

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

function NeuralNetworkModel:forwardPropagate(featureMatrix, saveTables, doNotDropoutNeurons)

	if (self.ModelParameters == nil) then self:generateLayers() end

	local ModelParameters = self.ModelParameters

	local numberOfLayers = #self.numberOfNeuronsTable

	local hasBiasNeuronTable = self.hasBiasNeuronTable

	local activationFunctionTable = self.activationFunctionTable

	local dropoutRateTable = self.dropoutRateTable

	local forwardPropagateTable = {}

	local zTable = {}
	
	local activationFunctionName = activationFunctionTable[1]
	
	local elementWiseActivationFunction = elementWiseActivationFunctionList[activationFunctionName]
	
	local layerZMatrix = featureMatrix

	local inputMatrix = featureMatrix

	local numberOfData = #featureMatrix
	
	if (elementWiseActivationFunction) then

		inputMatrix = AqwamMatrixLibrary:applyFunction(elementWiseActivationFunction, layerZMatrix)

	else

		inputMatrix = activationFunctionList[activationFunctionName](layerZMatrix)

	end

	inputMatrix = dropoutInputMatrix(inputMatrix, hasBiasNeuronTable[1], dropoutRateTable[1], doNotDropoutNeurons)

	table.insert(zTable, inputMatrix)

	table.insert(forwardPropagateTable, inputMatrix) -- don't remove this! otherwise the code won't work!

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local weightMatrix = ModelParameters[layerNumber]

		local hasBiasNeuron = hasBiasNeuronTable[layerNumber + 1]

		local activationFunctionName = activationFunctionTable[layerNumber + 1]

		local dropoutRate = dropoutRateTable[layerNumber + 1]
		
		local elementWiseActivationFunction = elementWiseActivationFunctionList[activationFunctionName]

		layerZMatrix = AqwamMatrixLibrary:dotProduct(inputMatrix, weightMatrix)

		if (typeof(layerZMatrix) == "number") then layerZMatrix = {{layerZMatrix}} end
		
		if (elementWiseActivationFunction) then

			inputMatrix = AqwamMatrixLibrary:applyFunction(elementWiseActivationFunction, layerZMatrix)

		else

			inputMatrix = activationFunctionList[activationFunctionName](layerZMatrix)

		end

		if (hasBiasNeuron == 1) then

			for data = 1, numberOfData, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron.

		end

		inputMatrix = dropoutInputMatrix(inputMatrix, hasBiasNeuron, dropoutRate, doNotDropoutNeurons)

		table.insert(zTable, layerZMatrix)

		table.insert(forwardPropagateTable, inputMatrix)

		self:sequenceWait()

	end

	if saveTables then

		self.forwardPropagateTable = forwardPropagateTable

		self.zTable = zTable

	end

	return inputMatrix, forwardPropagateTable, zTable

end

function NeuralNetworkModel:calculateCostFunctionDerivativeMatrixTable(lossMatrix)
	
	local forwardPropagateTable = self.forwardPropagateTable

	local zTable = self.zTable
	
	if (forwardPropagateTable == nil) then error("Table not found for forward propagation.") end

	if (zTable == nil) then error("Table not found for z matrix.") end

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local costFunctionDerivativeMatrixTable = {}

	local errorMatrixTable = {}
	
	local numberOfData = #lossMatrix

	local ModelParameters = self.ModelParameters

	local numberOfLayers = #self.numberOfNeuronsTable

	local activationFunctionTable = self.activationFunctionTable

	local hasBiasNeuronTable = self.hasBiasNeuronTable
	
	local activationFunctionName = activationFunctionTable[numberOfLayers]
	
	local elementWiseActivationFunctionDerivative = elementWiseActivationFunctionDerivativeList[activationFunctionName]
	
	local lastActivationMatrix = forwardPropagateTable[numberOfLayers]
	
	local lastZMatrix = zTable[numberOfLayers]
	
	local derivativeMatrix
	
	if (elementWiseActivationFunctionDerivative) then
		
		derivativeMatrix = AqwamMatrixLibrary:applyFunction(elementWiseActivationFunctionDerivative, lastZMatrix)
		
	else
		
		derivativeMatrix = activationFunctionDerivativeList[activationFunctionName](lastActivationMatrix, lastZMatrix)
		
	end

	local layerCostMatrix = AqwamMatrixLibrary:multiply(lossMatrix, derivativeMatrix)

	table.insert(errorMatrixTable, layerCostMatrix)

	for layerNumber = (numberOfLayers - 1), 2, -1 do

		activationFunctionName = activationFunctionTable[layerNumber]

		local hasBiasNeuronOnNextLayer = hasBiasNeuronTable[layerNumber + 1]

		local layerMatrix = AqwamMatrixLibrary:transpose(ModelParameters[layerNumber])

		local partialErrorMatrix = AqwamMatrixLibrary:dotProduct(layerCostMatrix, layerMatrix)
		
		local elementWiseActivationFunctionDerivative = elementWiseActivationFunctionDerivativeList[activationFunctionName]

		local currentAxtivationMatrix = forwardPropagateTable[layerNumber]

		local currentZMatrix = zTable[layerNumber]

		local derivativeMatrix

		if (elementWiseActivationFunctionDerivative) then

			derivativeMatrix = AqwamMatrixLibrary:applyFunction(elementWiseActivationFunctionDerivative, lastZMatrix)

		else

			derivativeMatrix = activationFunctionDerivativeList[activationFunctionName](currentAxtivationMatrix, currentZMatrix)

		end

		if (hasBiasNeuronOnNextLayer == 1) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

			for i = 1, #derivativeMatrix, 1 do derivativeMatrix[i][1] = 0 end

		end

		layerCostMatrix = AqwamMatrixLibrary:multiply(partialErrorMatrix, derivativeMatrix)

		table.insert(errorMatrixTable, 1, layerCostMatrix)

		self:sequenceWait()

	end

	for layer = 1, (numberOfLayers - 1), 1 do

		local activationLayerMatrix = AqwamMatrixLibrary:transpose(forwardPropagateTable[layer])

		local errorMatrix = errorMatrixTable[layer]

		local costFunctionDerivatives = AqwamMatrixLibrary:dotProduct(activationLayerMatrix, errorMatrix)

		if (type(costFunctionDerivatives) == "number") then costFunctionDerivatives = {{costFunctionDerivatives}} end

		table.insert(costFunctionDerivativeMatrixTable, costFunctionDerivatives)

		self:sequenceWait()

	end

	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivativeMatrixTable end

	return costFunctionDerivativeMatrixTable

end

function NeuralNetworkModel:gradientDescent(costFunctionDerivativeMatrixTable, numberOfData)

	local NewModelParameters = {}

	local numberOfLayers = #self.numberOfNeuronsTable

	local learningRateTable = self.learningRateTable

	local OptimizerTable = self.OptimizerTable

	local RegularizerTable = self.RegularizerTable

	local hasBiasNeuronTable = self.hasBiasNeuronTable

	local ModelParameters = self.ModelParameters

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local learningRate = learningRateTable[layerNumber + 1]

		local Regularizer = RegularizerTable[layerNumber + 1]

		local Optimizer = OptimizerTable[layerNumber + 1]

		local costFunctionDerivativeMatrix = costFunctionDerivativeMatrixTable[layerNumber]

		local hasBiasNeuronOnNextLayer = hasBiasNeuronTable[layerNumber + 1]

		if (type(costFunctionDerivativeMatrix) == "number") then costFunctionDerivativeMatrix = {{costFunctionDerivativeMatrix}} end

		local weightMatrix = ModelParameters[layerNumber]

		if (Regularizer ~= 0) then

			local regularizationDerivativeMatrix = Regularizer:calculateRegularizationDerivatives(weightMatrix)

			costFunctionDerivativeMatrix = AqwamMatrixLibrary:add(costFunctionDerivativeMatrix, regularizationDerivativeMatrix)

		end

		costFunctionDerivativeMatrix = AqwamMatrixLibrary:divide(costFunctionDerivativeMatrix, numberOfData)

		if (Optimizer ~= 0) then

			costFunctionDerivativeMatrix = Optimizer:calculate(learningRate, costFunctionDerivativeMatrix)

		else

			costFunctionDerivativeMatrix = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivativeMatrix)

		end

		local newWeightMatrix = AqwamMatrixLibrary:subtract(weightMatrix, costFunctionDerivativeMatrix)

		if (hasBiasNeuronOnNextLayer == 1) then -- There are two bias here, one for previous layer and one for the next one. In order the previous values does not propagate to the next layer, the first column must be set to zero, since the first column refers to bias for next layer. The first row is for bias at the current layer.

			for i = 1, #newWeightMatrix, 1 do newWeightMatrix[i][1] = 0 end

		end

		table.insert(NewModelParameters, newWeightMatrix)

	end

	return NewModelParameters

end

function NeuralNetworkModel:backwardPropagate(lossMatrix, clearTables)

	if (type(lossMatrix) == "number") then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local costFunctionDerivativeMatrixTable = self:calculateCostFunctionDerivativeMatrixTable(lossMatrix)

	self.ModelParameters = self:gradientDescent(costFunctionDerivativeMatrixTable, numberOfData)

	if (clearTables) then

		self.forwardPropagateTable = nil

		self.zTable = nil

	end

end

function NeuralNetworkModel:calculateCost(allOutputsMatrix, logisticMatrix, numberOfData)

	local subtractedMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local squaredSubtractedMatrix = AqwamMatrixLibrary:power(subtractedMatrix, 2)

	local totalCost = AqwamMatrixLibrary:sum(squaredSubtractedMatrix)

	local numberOfLayers = #self.numberOfNeuronsTable

	local RegularizerTable = self.RegularizerTable

	local ModelParameters = self.ModelParameters

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local Regularizer = RegularizerTable[layerNumber + 1]

		if (Regularizer ~=  0) then totalCost = totalCost + Regularizer:calculateRegularization(ModelParameters[layerNumber]) end

	end

	local cost = totalCost / numberOfData

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

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValue(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue

end

function NeuralNetworkModel:getLabelFromOutputMatrix(outputMatrix)

	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	local predictedLabelVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValueVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

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

function NeuralNetworkModel.new(maximumNumberOfIterations)

	local NewNeuralNetworkModel = GradientMethodBaseModel.new()

	setmetatable(NewNeuralNetworkModel, NeuralNetworkModel)

	NewNeuralNetworkModel.maximumNumberOfIterations = maximumNumberOfIterations or defaultMaximumNumberOfIterations

	NewNeuralNetworkModel.numberOfNeuronsTable = {}

	NewNeuralNetworkModel.RegularizerTable = {}

	NewNeuralNetworkModel.OptimizerTable = {}

	NewNeuralNetworkModel.ClassesList = {}

	NewNeuralNetworkModel.hasBiasNeuronTable = {}

	NewNeuralNetworkModel.learningRateTable = {}

	NewNeuralNetworkModel.activationFunctionTable = {}

	NewNeuralNetworkModel.dropoutRateTable = {}

	return NewNeuralNetworkModel

end

function NeuralNetworkModel:setParameters(maximumNumberOfIterations)

	self.maximumNumberOfIterations = maximumNumberOfIterations or self.maximumNumberOfIterations

end

function NeuralNetworkModel:generateLayers()

	local layersArray = self.numberOfNeuronsTable

	local numberOfLayers = #layersArray

	if (#self.numberOfNeuronsTable == 1) then error("There is only one layer!") end

	local ModelParameters = {}

	for layer = 1, (numberOfLayers - 1), 1 do

		local numberOfCurrentLayerNeurons = layersArray[layer]

		if (self.hasBiasNeuronTable[layer] == 1) then numberOfCurrentLayerNeurons += 1 end -- 1 is added for bias

		local numberOfNextLayerNeurons = layersArray[layer + 1]

		local hasBiasNeuronOnNextLayer = self.hasBiasNeuronTable[layer + 1] 

		if (hasBiasNeuronOnNextLayer == 1) then numberOfNextLayerNeurons += 1 end

		local weightMatrix = self:initializeMatrixBasedOnMode(numberOfCurrentLayerNeurons, numberOfNextLayerNeurons, 0, hasBiasNeuronOnNextLayer) -- Since no outputs are going into the bias neuron, it should not be considered as an input neuron. So the bias column needed to be excluded for our weight initialization.

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

	self.numberOfNeuronsTable = numberOfNeuronsArray

	self.hasBiasNeuronTable = {}

	self.learningRateTable = {}

	self.activationFunctionTable = {}

	self.dropoutRateTable = {}

	self.OptimizerTable = {}

	self.RegularizerTable = {}

	local numberOfLayers = #self.numberOfNeuronsTable

	for layer = 1, numberOfLayers, 1 do

		self.activationFunctionTable[layer] = activationFunction

		self.learningRateTable[layer] = learningRate

		self.dropoutRateTable[layer] = dropoutRate

		self.hasBiasNeuronTable[layer] = ((layer == numberOfLayers) and 0) or 1

		self.OptimizerTable[layer] = OptimizerArray[layer] or 0

		self.RegularizerTable[layer] = RegularizerArray[layer] or 0

	end

	self:generateLayers()

end

function NeuralNetworkModel:addLayer(numberOfNeurons, hasBiasNeuron, activationFunction, learningRate, Optimizer, Regularizer, dropoutRate)

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

	table.insert(self.numberOfNeuronsTable, numberOfNeurons)

	table.insert(self.hasBiasNeuronTable, hasBiasNeuron)

	table.insert(self.activationFunctionTable, activationFunction)

	table.insert(self.learningRateTable, learningRate)

	table.insert(self.OptimizerTable, Optimizer or 0)

	table.insert(self.RegularizerTable, Regularizer or 0)

	table.insert(self.dropoutRateTable, dropoutRate)

end

function NeuralNetworkModel:setLayer(layerNumber, hasBiasNeuron, activationFunction, learningRate, Optimizer, Regularizer, dropoutRate)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsTable)  then

		error("The layer number exceeds the number of layers!") 

	end 

	layerPropertyValueTypeCheckingFunctionList["HasBias"](hasBiasNeuron)

	layerPropertyValueTypeCheckingFunctionList["ActivationFunction"](activationFunction)

	layerPropertyValueTypeCheckingFunctionList["LearningRate"](learningRate)

	layerPropertyValueTypeCheckingFunctionList["DropoutRate"](dropoutRate)

	hasBiasNeuron = self:getValueOrDefaultValue(hasBiasNeuron,  self.hasBiasNeuronTable[layerNumber])

	hasBiasNeuron = (hasBiasNeuron and 1) or 0

	Regularizer = self:getValueOrDefaultValue(Regularizer,  self.RegularizerTable[layerNumber])

	Regularizer = Regularizer or 0

	Optimizer = self:getValueOrDefaultValue(Optimizer,  self.OptimizerTable[layerNumber])

	Optimizer = Optimizer or 0

	self.hasBiasNeuronTable[layerNumber] = hasBiasNeuron

	self.activationFunctionTable[layerNumber] = activationFunction or self.activationFunctionTable[layerNumber] 

	self.learningRateTable[layerNumber] = activationFunction or self.learningRateTable[layerNumber] 

	self.OptimizerTable[layerNumber] = Optimizer

	self.RegularizerTable[layerNumber] = Regularizer

	self.dropoutRateTable[layerNumber] = dropoutRate or self.dropoutRateTable[layerNumber]

end

function NeuralNetworkModel:setLayerProperty(layerNumber, property, value)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsTable)  then

		error("The layer number exceeds the number of layers!") 

	end 

	if (property == "HasBias") then

		layerPropertyValueTypeCheckingFunctionList["HasBias"](value)

		local hasBiasNeuron = self:getValueOrDefaultValue(value,  self.hasBiasNeuronTable[layerNumber])

		hasBiasNeuron = (hasBiasNeuron and 1) or 0

		self.hasBiasNeuronTable[layerNumber] = hasBiasNeuron

	elseif (property == "ActivationFunction") then

		layerPropertyValueTypeCheckingFunctionList["ActivationFunction"](value)

		self.activationFunctionTable[layerNumber] = value or self.activationFunctionTable[layerNumber]

	elseif (property == "LearningRate") then

		layerPropertyValueTypeCheckingFunctionList["LearningRate"](value)

		self.learningRateTable[layerNumber] = value or self.learningRateTable[layerNumber]

	elseif (property == "Optimizer") then

		value = self:getValueOrDefaultValue(value, self.OptimizerTable[layerNumber])

		value = value or 0

		self.OptimizerTable[layerNumber] = value

	elseif (property == "Regularizer") then

		value = self:getValueOrDefaultValue(value, self.OptimizerTable[layerNumber])

		value = value or 0

		self.RegularizerTable[layerNumber] = value or 0

	elseif (property == "DropoutRate") then

		layerPropertyValueTypeCheckingFunctionList["DropoutRate"](value)

		self.dropoutRateTable[layerNumber] = value or self.dropoutRateTable[layerNumber]

	else

		warn("Layer property does not exists. Did not change the layer's properties.")

	end

end

function NeuralNetworkModel:getLayerProperty(layerNumber, property)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsTable)  then

		error("The layer number exceeds the number of layers!") 

	end 

	if (property == "HasBias") then

		return (self.hasBiasNeuronTable[layerNumber] == 1)

	elseif (property == "ActivationFunction") then

		return self.activationFunctionTable[layerNumber]

	elseif (property == "LearningRate") then

		return self.learningRateTable[layerNumber]

	elseif (property == "Optimizer") then

		local Optimizer = self.OptimizerTable[layerNumber]

		if (Optimizer ~= 0) then

			return Optimizer

		else

			return nil

		end

	elseif (property == "Regularizer") then

		local Regularizer = self.RegularizerTable[layerNumber]

		if (Regularizer ~= 0) then

			return Regularizer

		else

			return nil

		end

	elseif (property == "DropoutRate") then

		return self.dropoutRateTable[layerNumber]

	else

		warn("Layer property does not exists. Returning nil value.")

		return nil

	end

end

function NeuralNetworkModel:getLayer(layerNumber)

	if (layerNumber <= 0) then 

		error("The layer number can't be less than or equal to zero!") 

	elseif (layerNumber > #self.numberOfNeuronsTable) then

		error("The layer number exceeds the number of layers!") 

	end 

	local Optimizer = self.OptimizerTable[layerNumber]

	if (Optimizer == 0) then

		Optimizer = nil

	end

	local Regularizer = self.RegularizerTable[layerNumber]

	if (Regularizer == 0) then

		Regularizer = nil

	end

	return self.numberOfNeuronsTable[layerNumber], (self.hasBiasNeuronTable[layerNumber] == 1), self.activationFunctionTable[layerNumber], self.learningRateTable[layerNumber], Optimizer, Regularizer, self.dropoutRateTable[layerNumber]

end

function NeuralNetworkModel:getTotalNumberOfNeurons(layerNumber)

	return self.numberOfNeuronsTable[layerNumber] + self.hasBiasNeuronTable[layerNumber]

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

		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrixLeft, currentWeightMatrixToAdd, currentWeightMatrixRight)
		newNextWeightMatrix = AqwamMatrixLibrary:verticalConcatenate(nextWeightMatrixTop, nextWeightMatrixToAdd, nextWeightMatrixBottom)

	else

		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrixLeft, currentWeightMatrixRight, currentWeightMatrixToAdd)
		newNextWeightMatrix = AqwamMatrixLibrary:verticalConcatenate(nextWeightMatrixTop, nextWeightMatrixBottom, nextWeightMatrixToAdd)

	end

	return newCurrentWeightMatrix, newNextWeightMatrix

end

function NeuralNetworkModel:evolveLayerSize(layerNumber, initialNeuronIndex, size)

	if (self.ModelParameters == nil) then error("No Model Parameters!") end

	if (#self.ModelParameters == 0) then 

		self.ModelParameters = nil
		error("No Model Parameters!") 

	end

	local numberOfLayers = #self.numberOfNeuronsTable -- DON'T FORGET THAT IT DOES NOT INCLUDE BIAS!

	if (layerNumber > numberOfLayers) then error("Layer number exceeds this model's number of layers.") end

	local hasBiasNeuronValue = self.hasBiasNeuronTable[layerNumber]

	local numberOfNeurons = self.numberOfNeuronsTable[layerNumber] + hasBiasNeuronValue

	local currentWeightMatrix
	local nextWeightMatrix

	if (layerNumber == numberOfLayers) then

		currentWeightMatrix = self.ModelParameters[numberOfLayers - 1]

	elseif (layerNumber > 1) and (layerNumber < numberOfLayers) then

		currentWeightMatrix = self.ModelParameters[layerNumber - 1]
		nextWeightMatrix = self.ModelParameters[layerNumber]

	else

		currentWeightMatrix = self.ModelParameters[1]
		nextWeightMatrix = self.ModelParameters[2]

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

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode(#currentWeightMatrix, size)
		nextWeightMatrixToAdd =  self:initializeMatrixBasedOnMode(size, #nextWeightMatrix[1])

		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrix, currentWeightMatrixToAdd)
		newNextWeightMatrix = AqwamMatrixLibrary:verticalConcatenate(nextWeightMatrix, nextWeightMatrixToAdd)

	elseif (initialNeuronIndex == 0) and (size > 0) and (not hasNextLayer) then

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode(#currentWeightMatrix, size)
		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrixToAdd, currentWeightMatrix)

	elseif (initialNeuronIndex > 0) and (size > 0) and (hasNextLayer) then

		currentWeightMatrixLeft = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, 1, initialNeuronIndex)
		currentWeightMatrixRight = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, initialNeuronIndex + 1, #currentWeightMatrix[1])

		nextWeightMatrixTop = AqwamMatrixLibrary:extractRows(nextWeightMatrix, 1, initialNeuronIndex)
		nextWeightMatrixBottom = AqwamMatrixLibrary:extractRows(nextWeightMatrix, initialNeuronIndex + 1, #nextWeightMatrix)

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode(#currentWeightMatrix, size)
		nextWeightMatrixToAdd =  self:initializeMatrixBasedOnMode(size, #nextWeightMatrix[1])

		newCurrentWeightMatrix, newNextWeightMatrix = mergeLayers(numberOfNeurons, initialNeuronIndex, currentWeightMatrixLeft, currentWeightMatrixRight, currentWeightMatrixToAdd, nextWeightMatrixTop, nextWeightMatrixToAdd, nextWeightMatrixBottom)

	elseif (initialNeuronIndex > 0) and (size > 0) and (not hasNextLayer) then

		currentWeightMatrixToAdd = self:initializeMatrixBasedOnMode(#currentWeightMatrix, size)
		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrix, currentWeightMatrixToAdd)

	elseif (size == -1) and (hasNextLayer) and (numberOfNeurons == 1) then

		newCurrentWeightMatrix = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, initialNeuronIndex, initialNeuronIndex)
		newNextWeightMatrix = AqwamMatrixLibrary:extractRows(nextWeightMatrix, initialNeuronIndex, initialNeuronIndex)

	elseif (size == -1) and (not hasNextLayer) and (numberOfNeurons == 1) then

		newCurrentWeightMatrix = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, initialNeuronIndex, initialNeuronIndex)

	elseif (size < 0) and (hasNextLayer) and (numberOfNeurons >= absoluteSize) then

		currentWeightMatrixLeft = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, 1, secondNeuronIndex)
		currentWeightMatrixRight = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, thirdNeuronIndex, #currentWeightMatrix[1])

		nextWeightMatrixTop = AqwamMatrixLibrary:extractRows(nextWeightMatrix, 1, secondNeuronIndex)
		nextWeightMatrixBottom = AqwamMatrixLibrary:extractRows(nextWeightMatrix, thirdNeuronIndex, #nextWeightMatrix)

		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrixLeft, currentWeightMatrixRight)
		newNextWeightMatrix = AqwamMatrixLibrary:verticalConcatenate(nextWeightMatrixTop, nextWeightMatrixBottom)

	elseif (size < 0) and (not hasNextLayer) and (numberOfNeurons >= absoluteSize) then

		currentWeightMatrixLeft = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, 1, secondNeuronIndex)
		currentWeightMatrixRight = AqwamMatrixLibrary:extractColumns(currentWeightMatrix, thirdNeuronIndex, #currentWeightMatrix[1])

		newCurrentWeightMatrix = AqwamMatrixLibrary:horizontalConcatenate(currentWeightMatrixLeft, currentWeightMatrixRight)

	end

	if (layerNumber == numberOfLayers) then

		self.ModelParameters[numberOfLayers - 1] = newCurrentWeightMatrix

	elseif (layerNumber > 1) and (layerNumber < numberOfLayers) then

		self.ModelParameters[layerNumber - 1] = newCurrentWeightMatrix
		self.ModelParameters[layerNumber] = newNextWeightMatrix

	else

		self.ModelParameters[1] = newCurrentWeightMatrix
		self.ModelParameters[2] = newNextWeightMatrix

	end

	self.numberOfNeuronsTable[layerNumber] += size

end

function NeuralNetworkModel:train(featureMatrix, labelVector)

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local numberOfNeuronsAtInputLayer = self.numberOfNeuronsTable[1] + self.hasBiasNeuronTable[1]

	if (numberOfNeuronsAtInputLayer ~= numberOfFeatures) then error("Input layer has " .. numberOfNeuronsAtInputLayer .. " neuron(s), but feature matrix has " .. #featureMatrix[1] .. " features!") end

	if (#featureMatrix ~= #labelVector) then error("Number of rows of feature matrix and the label vector is not the same!") end

	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	local numberOfIterations = 0

	local cost

	local costArray = {}

	local deltaTable

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

		numberOfIterations += 1

		self:iterationWait()

		activatedOutputsMatrix = self:forwardPropagate(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(activatedOutputsMatrix, logisticMatrix, numberOfData)

		end)

		if cost then 

			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)

		end

		local lossMatrix = AqwamMatrixLibrary:subtract(activatedOutputsMatrix, logisticMatrix)

		lossMatrix = AqwamMatrixLibrary:divide(lossMatrix, numberOfData)

		self:backwardPropagate(lossMatrix, true)

	until (numberOfIterations == self.maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	if (self.autoResetOptimizers) then

		for i, Optimizer in ipairs(self.OptimizerTable) do

			if (Optimizer ~= 0) then Optimizer:reset() end

		end

	end

	return costArray

end

function NeuralNetworkModel:reset()

	for i, Optimizer in ipairs(self.OptimizerTable) do

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

	for i = 1, #self.numberOfNeuronsTable do

		maxLayerLength = math.max(maxLayerLength, string.len(tostring(i)))

		maxNeuronsLength = math.max(maxNeuronsLength, string.len(tostring(self.numberOfNeuronsTable[i])))

		hasBias = (self.hasBiasNeuronTable[i] == 1)

		maxBiasLength = math.max(maxBiasLength, string.len(tostring(hasBias)))

		maxActivationLength = math.max(maxActivationLength, string.len(self.activationFunctionTable[i]))

		maxLearningRateLength = math.max(maxLearningRateLength, string.len(tostring(self.learningRateTable[i])))

		maxOptimizerLength = math.max(maxOptimizerLength, string.len("false"))

		maxRegularizerLength = math.max(maxRegularizerLength, string.len("false"))

		maxDropoutRateLength = math.max(maxDropoutRateLength, string.len(tostring(self.dropoutRateTable[i])))

	end

	-- Print the table header

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
	for i = 1, #self.numberOfNeuronsTable do

		local layer = "| " .. string.format("%-" .. maxLayerLength .. "s", i) .. " "

		local neurons = "| " .. string.format("%-" .. maxNeuronsLength .. "s", self.numberOfNeuronsTable[i]) .. " "

		hasBias = (self.hasBiasNeuronTable[i] == 1)

		local bias = "| " .. string.format("%-" .. maxBiasLength .. "s", tostring(hasBias)) .. " "

		local activation = "| " .. string.format("%-" .. maxActivationLength .. "s", self.activationFunctionTable[i]) .. " "

		local learningRate = "| " .. string.format("%-" .. maxLearningRateLength .. "s", self.learningRateTable[i]) .. " "

		local optimizer = "| " .. string.format("%-" .. maxOptimizerLength .. "s", self.OptimizerTable[i] and "true" or "false") .. " "

		local regularization = "| " .. string.format("%-" .. maxRegularizerLength .. "s", self.RegularizerTable[i] and "true" or "false") .. " "

		local dropoutRate = "| " .. string.format("%-" .. maxDropoutRateLength .. "s", self.dropoutRateTable[i]) .. " |"

		local stringPart = layer .. neurons .. bias .. activation .. learningRate .. optimizer .. regularization .. dropoutRate .. "\n"

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

	return #self.numberOfNeuronsTable

end

return NeuralNetworkModel