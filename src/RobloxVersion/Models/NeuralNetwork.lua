local BaseModel = require(script.Parent.BaseModel)

NeuralNetworkModel = {}

NeuralNetworkModel.__index = NeuralNetworkModel

setmetatable(NeuralNetworkModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultActivationFunction = "LeakyReLU"

local defaultTargetCost = 0

local activationFunctionList = {

	["Sigmoid"] = function (zMatrix) 

		local sigmoidFunction = function(z) return 1/(1 + math.exp(-1 * z)) end

		local aMatrix = AqwamMatrixLibrary:applyFunction(sigmoidFunction, zMatrix)

		return aMatrix

	end,

	["Tanh"] = function (zMatrix) 

		local aMatrix = AqwamMatrixLibrary:applyFunction(math.tanh, zMatrix)

		return aMatrix

	end,

	["ReLU"] = function (zMatrix) 

		local ReLUFunction = function (z) return math.max(0, z) end

		local aMatrix = AqwamMatrixLibrary:applyFunction(ReLUFunction, zMatrix)

		return aMatrix

	end,

	["LeakyReLU"] = function (zMatrix) 

		local LeakyReLU = function (z) return math.max((0.01 * z), z) end

		local aMatrix = AqwamMatrixLibrary:applyFunction(LeakyReLU, zMatrix)

		return aMatrix

	end,

	["ELU"] = function (zMatrix) 

		local ELUFunction = function (z) return if (z > 0) then z else (0.01 * (math.exp(z) - 1)) end

		local aMatrix = AqwamMatrixLibrary:applyFunction(ELUFunction, zMatrix)

		return aMatrix

	end,
	
	["Gaussian"] = function (zMatrix)

		local GaussianFunction = function (z) return math.exp(-math.pow(z, 2)) end

		local aMatrix = AqwamMatrixLibrary:applyFunction(GaussianFunction, zMatrix)

		return aMatrix

	end,

	["SiLU"] = function (zMatrix)

		local SiLUFunction = function (z) return z / (1 + math.exp(-z)) end

		local aMatrix = AqwamMatrixLibrary:applyFunction(SiLUFunction, zMatrix)

		return aMatrix

	end,
	
	["Mish"] = function (zMatrix)
		
		local MishFunction = function (z) return z * math.tanh(math.log(1 + math.exp(z))) end
		
		local aMatrix = AqwamMatrixLibrary:applyFunction(MishFunction, zMatrix)
		
		return aMatrix
		
	end,

	["BinaryStep"] = function (aMatrix, zMatrix)

		local BinaryStepFunction = function (z) return ((z > 0) and 1) or 0 end

		local aMatrix = AqwamMatrixLibrary:applyFunction(BinaryStepFunction, zMatrix)

		return aMatrix

	end,

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

			local highestZValue = AqwamMatrixLibrary:findMaximumValueInMatrix(zVector)

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

local derivativeList = {

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

	["ReLU"] = function (aMatrix, zMatrix)

		local ReLUDerivativeFunction = function (z) if (z > 0) then return 1 else return 0 end end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(ReLUDerivativeFunction, zMatrix)

		return derivativeMatrix

	end,

	["LeakyReLU"] = function (aMatrix, zMatrix)

		local LeakyReLUDerivativeFunction = function (z) if (z > 0) then return 1 else return 0.01 end end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(LeakyReLUDerivativeFunction, zMatrix)

		return derivativeMatrix

	end,

	["ELU"] = function (aMatrix, zMatrix)

		local ELUDerivativeFunction = function (z) if (z > 0) then return 1 else return 0.01 * math.exp(z) end end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(ELUDerivativeFunction, zMatrix)

		return derivativeMatrix


	end,
	
	["Gaussian"] = function (aMatrix, zMatrix)

		local GaussianDerivativeFunction = function (z) return -2 * z * math.exp(-math.pow(z, 2)) end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(GaussianDerivativeFunction, zMatrix)

		return derivativeMatrix

	end,

	["SiLU"] = function (aMatrix, zMatrix)

		local SiLUDerivativeFunction = function (z) return (1 + math.exp(-z) + (z * math.exp(-z))) / (1 + math.exp(-z))^2 end

		local derivativeMatrix = AqwamMatrixLibrary:applyFunction(SiLUDerivativeFunction, zMatrix)

		return derivativeMatrix

	end,
	
	["Mish"] = function (aMatrix, zMatrix)

		local MishDerivativeFunction = function (z) 
			
			return math.exp(z) * (math.exp(3 * z) + 4 * math.exp(2 * z) + (6 + 4 * z) * math.exp(z) + 4 * (1 + z)) / math.pow((1 + math.pow((math.exp(z) + 1), 2)), 2)
			
		end

		local aMatrix = AqwamMatrixLibrary:applyFunction(MishDerivativeFunction, zMatrix)

		return aMatrix

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

function NeuralNetworkModel:forwardPropagate(featureMatrix, saveTables)
	
	if (self.ModelParameters == nil) then self:generateLayers() end

	local layerZ

	local forwardPropagateTable = {}

	local zTable = {}

	local inputMatrix = featureMatrix

	local numberOfData = #featureMatrix

	local numberOfLayers = #self.numberOfNeuronsTable

	table.insert(zTable, inputMatrix)

	table.insert(forwardPropagateTable, inputMatrix) -- don't remove this! otherwise the code won't work!
	
	for layerNumber = 1, (numberOfLayers - 1), 1 do
		
		local weightMatrix = self.ModelParameters[layerNumber]
		
		local hasBiasNeuron = self.hasBiasNeuronTable[layerNumber]

		local activationFunctionName = self.activationFunctionTable[layerNumber]

		local activationFunction = activationFunctionList[activationFunctionName]

		layerZ = AqwamMatrixLibrary:dotProduct(inputMatrix, weightMatrix)

		if (typeof(layerZ) == "number") then layerZ = {{layerZ}} end
		
		if (hasBiasNeuron == 1) then

			for data = 1, numberOfData, 1 do layerZ[data][1] = 0 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to zero so that it remains as bias neuron.

		end

		inputMatrix = activationFunction(layerZ)

		if (hasBiasNeuron == 1) then

			for data = 1, numberOfData, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron.

		end

		table.insert(zTable, layerZ)

		table.insert(forwardPropagateTable, inputMatrix)
		
	end

	local activationFunctionName = self.activationFunctionTable[numberOfLayers]

	local activationFunction = activationFunctionList[activationFunctionName]

	inputMatrix = activationFunction(layerZ)

	table.insert(zTable, layerZ)

	table.insert(forwardPropagateTable, inputMatrix)

	if saveTables then

		self.forwardPropagateTable = forwardPropagateTable

		self.zTable = zTable

	end

	return inputMatrix

end

function NeuralNetworkModel:calculateErrorMatrix(lossMatrix, forwardPropagateTable, zTable)

	local errorMatrixTable = {}

	local numberOfLayers = #self.numberOfNeuronsTable

	local zLayerMatrix
	
	local activationFunctionName = self.activationFunctionTable[numberOfLayers]

	local derivativeFunction = derivativeList[activationFunctionName]
	
	local derivativeMatrix = derivativeFunction(forwardPropagateTable[numberOfLayers], zTable[numberOfLayers])

	local layerCostMatrix = AqwamMatrixLibrary:multiply(lossMatrix, derivativeMatrix)

	table.insert(errorMatrixTable, layerCostMatrix)

	for output = (numberOfLayers - 1), 2, -1 do

		local activationFunctionName = self.activationFunctionTable[output]

		local derivativeFunction = derivativeList[activationFunctionName]

		local layerMatrix = self.ModelParameters[output]

		local layerMatrix = AqwamMatrixLibrary:transpose(layerMatrix)

		local partialErrorMatrix = AqwamMatrixLibrary:dotProduct(layerCostMatrix, layerMatrix)

		local derivativeMatrix = derivativeFunction(forwardPropagateTable[output], zTable[output])

		layerCostMatrix = AqwamMatrixLibrary:multiply(partialErrorMatrix, derivativeMatrix)

		table.insert(errorMatrixTable, 1, layerCostMatrix)

	end

	return errorMatrixTable

end

function NeuralNetworkModel:calculateDelta(forwardPropagateTable, errorMatrixTable)

	local deltaTable = {}
	
	local numberOfLayers = #self.numberOfNeuronsTable

	for layer = 1, (numberOfLayers - 1), -1 do

		local activationLayerMatrix = AqwamMatrixLibrary:transpose(forwardPropagateTable[layer])

		local errorMatrix = errorMatrixTable[layer]

		local costFunctionDerivatives = AqwamMatrixLibrary:dotProduct(activationLayerMatrix, errorMatrix)

		if (type(costFunctionDerivatives) == "number") then costFunctionDerivatives = {{costFunctionDerivatives}} end

		table.insert(deltaTable, costFunctionDerivatives)

	end

	return deltaTable

end

function NeuralNetworkModel:calculateCostFunctionDerivatives(learningRate, deltaTable, numberOfData)
	
	local regularizationDerivatives

	local costFunctionDerivativesTable = {}

	local newWeightMatrix

	local numberOfLayers = #self.numberOfNeuronsTable

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local costFunctionDerivatives = deltaTable[layerNumber]

		local learningRate = self.learningRateTable[layerNumber]

		local Regularization = self.RegularizationTable[layerNumber]

		local Optimizer = self.OptimizerTable[layerNumber]

		local weightMatrix = self.ModelParameters[layerNumber]

		local calculatedLearningRate = learningRate / numberOfData

		if Regularization then

			regularizationDerivatives = Regularization:calculateRegularizationDerivatives(weightMatrix, numberOfData)

			costFunctionDerivatives = AqwamMatrixLibrary:add(costFunctionDerivatives, regularizationDerivatives)

		end

		if Optimizer then

			costFunctionDerivatives = Optimizer:calculate(calculatedLearningRate, costFunctionDerivatives)

		else

			costFunctionDerivatives = AqwamMatrixLibrary:multiply(calculatedLearningRate, costFunctionDerivatives)

		end

		table.insert(costFunctionDerivativesTable, costFunctionDerivatives)

	end
	
	return costFunctionDerivativesTable
	
end

function NeuralNetworkModel:gradientDescent(costFunctionDerivativesTable)

	local NewModelParameters = {}

	local numberOfLayers = #self.numberOfNeuronsTable

	for layerNumber = 1, (numberOfLayers - 1), 1 do

		local weightMatrix = self.ModelParameters[layerNumber]
		
		local costFunctionDerivatives = costFunctionDerivativesTable[layerNumber]

		local newWeightMatrix = AqwamMatrixLibrary:subtract(weightMatrix, costFunctionDerivatives) 

		table.insert(NewModelParameters, newWeightMatrix)

	end

	return NewModelParameters

end

function NeuralNetworkModel:calculateCost(allOutputsMatrix, logisticMatrix)

	local numberOfData = #logisticMatrix

	local subtractedMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local squaredSubtractedMatrix = AqwamMatrixLibrary:power(subtractedMatrix, 2)

	local sumSquaredSubtractedMatrix = AqwamMatrixLibrary:sum(squaredSubtractedMatrix)

	local cost = sumSquaredSubtractedMatrix / numberOfData

	return cost

end

function NeuralNetworkModel:fetchValueFromScalar(outputVector)

	local value = outputVector[1][1]

	local numberOfLayers = #self.numberOfNeuronsTable

	local activationFunctionAtFinalLayer = self:getActivationLayerAtFinalLayer()

	local isValueOverCutOff = cutOffListForScalarValues[activationFunctionAtFinalLayer](value)

	local classIndex = (isValueOverCutOff and 2) or 1

	local predictedLabel = self.ClassesList[classIndex]

	return predictedLabel, value

end

function NeuralNetworkModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

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

function NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	local NewNeuralNetworkModel = BaseModel.new()

	setmetatable(NewNeuralNetworkModel, NeuralNetworkModel)

	NewNeuralNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewNeuralNetworkModel.learningRate = learningRate or defaultLearningRate

	NewNeuralNetworkModel.targetCost = targetCost or defaultTargetCost

	NewNeuralNetworkModel.numberOfNeuronsTable = {}

	NewNeuralNetworkModel.RegularizationTable = {}

	NewNeuralNetworkModel.OptimizerTable = {}

	NewNeuralNetworkModel.ClassesList = {}

	NewNeuralNetworkModel.hasBiasNeuronTable = {}

	NewNeuralNetworkModel.learningRateTable = {}

	NewNeuralNetworkModel.activationFunctionTable = {}

	NewNeuralNetworkModel.previousDeltaMatricesTable = {}

	return NewNeuralNetworkModel

end

function NeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

end

function NeuralNetworkModel:generateLayers()

	local layersArray = self.numberOfNeuronsTable

	local numberOfLayers = #layersArray

	if (#self.numberOfNeuronsTable == 1) then error("There is only one layer!") end

	local ModelParameters = {}

	local weightMatrix

	local numberOfCurrentLayerNeurons

	local numberOfNextLayerNeurons

	for layer = 1, (numberOfLayers - 1), 1 do

		numberOfCurrentLayerNeurons = layersArray[layer]

		if (self.hasBiasNeuronTable[layer] == 1) then numberOfCurrentLayerNeurons += 1 end -- 1 is added for bias

		numberOfNextLayerNeurons = layersArray[layer + 1]

		if (self.hasBiasNeuronTable[layer + 1] == 1) then numberOfNextLayerNeurons += 1 end

		weightMatrix = self:initializeMatrixBasedOnMode(numberOfCurrentLayerNeurons, numberOfNextLayerNeurons)

		table.insert(ModelParameters, weightMatrix)

	end

	self.ModelParameters = ModelParameters

end

function NeuralNetworkModel:createLayers(numberOfNeuronsArray, activationFunction, learningRate, Optimizer, Regularization)

	activationFunction = activationFunction or defaultActivationFunction

	learningRate = activationFunction or self.learningRate

	if (typeof(numberOfNeuronsArray) ~= "table") then error("Invalid input for number of neurons!") end

	if (typeof(activationFunction) ~= "string") then error("Invalid input for activation function!") end

	self.ModelParameters = nil

	self.numberOfNeuronsTable = numberOfNeuronsArray

	self.hasBiasNeuronTable = {}

	self.learningRateTable = {}

	self.activationFunctionTable = {}

	self.OptimizerTable = {}

	self.RegularizationTable = {}

	local numberOfLayers = #self.numberOfNeuronsTable

	for layer = 1, numberOfLayers, 1 do

		self.activationFunctionTable[layer] = activationFunction

		self.learningRateTable[layer] = learningRate

		if (layer == numberOfLayers) then

			self.hasBiasNeuronTable[layer] = 0

			self.OptimizerTable[layer] = Optimizer

			self.RegularizationTable[layer] = Regularization

		else

			self.hasBiasNeuronTable[layer] = 1

			self.OptimizerTable[layer] = nil

			self.RegularizationTable[layer] = nil

		end

	end

end

function NeuralNetworkModel:addLayer(numberOfNeurons, hasBiasNeuron, activationFunction, learningRate, Optimizer, Regularization)

	if (typeof(numberOfNeurons) ~= "number") then error("Invalid input for number of neurons!") end

	local hasBiasNeuronType = typeof(hasBiasNeuron)

	if (hasBiasNeuronType ~= "nil") and (hasBiasNeuronType ~= "boolean") then error("Invalid input for adding bias!") end

	local learningRateType = typeof(learningRate)

	if (learningRateType ~= "nil") and (learningRateType ~= "number") then error("Invalid input for learning rate!") end

	if (typeof(numberOfNeurons) ~= "number") then error("Invalid input for number of neurons!") end

	local activationFunctionType = typeof(activationFunction)

	if (activationFunctionType ~= "nil") and (activationFunctionType ~= "string") then error("Invalid input for activation function!") end

	hasBiasNeuron = self:getBooleanOrDefaultOption(hasBiasNeuron, true)
	
	hasBiasNeuron = (hasBiasNeuron and 1) or 0

	learningRate = learningRate or self.learningRate

	activationFunction = activationFunction or defaultActivationFunction

	table.insert(self.numberOfNeuronsTable, numberOfNeurons)

	table.insert(self.hasBiasNeuronTable, hasBiasNeuron)

	table.insert(self.activationFunctionTable, activationFunction)

	table.insert(self.learningRateTable, learningRate)

	table.insert(self.OptimizerTable, Optimizer)

	table.insert(self.RegularizationTable, Regularization)

end

function NeuralNetworkModel:setLayer(layerNumber, hasBiasNeuron, activationFunction, learningRate, Optimizer, Regularization)

	if (typeof(layerNumber) ~= "number") then error("Invalid input layer number!") end

	if (typeof(hasBiasNeuron) ~= "boolean") then error("Invalid input for adding bias!") end

	if (typeof(learningRate) ~= "number") then error("Invalid learning rate!") end

	if  (typeof(activationFunction) ~= "string") then error("Invalid input for activation function!") end 
	
	hasBiasNeuron = self:getBooleanOrDefaultOption(hasBiasNeuron,  self.hasBiasNeuronTable[layerNumber])
	
	hasBiasNeuron = (hasBiasNeuron and 1) or 0

	self.hasBiasNeuronTable[layerNumber] = hasBiasNeuron

	self.activationFunctionTable[layerNumber] = activationFunction or self.activationFunctionTable[layerNumber] 

	self.learningRateTable[layerNumber] = activationFunction or self.learningRateTable[layerNumber] 

	self.OptimizerTable[layerNumber] = Optimizer or self.OptimizerTable[layerNumber]

	self.RegularizationTable[layerNumber] = Regularization or self.RegularizationTable[layerNumber]

end

function NeuralNetworkModel:getLayer(layerNumber)

	return self.numberOfNeuronsTable[layerNumber], self.hasBiasNeuronTable[layerNumber], self.activationFunctionTable[layerNumber], self.learningRateTable[layerNumber], self.OptimizerTable[layerNumber], self.RegularizationTable[layerNumber]

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

function NeuralNetworkModel:backPropagate(lossMatrix, clearTables, doNotUpdateModelParameters)

	if (self.forwardPropagateTable == nil) then error("Table not found for forward propagation.") end

	if (self.zTable == nil) then error("Table not found for z matrix.") end

	if type(lossMatrix) == "number" then lossMatrix = {{lossMatrix}} end

	local numberOfData = #lossMatrix

	local errorMatrixTable = self:calculateErrorMatrix(lossMatrix, self.forwardPropagateTable, self.zTable)

	local deltaTable = self:calculateDelta(self.forwardPropagateTable, errorMatrixTable)
	
	local costFunctionDerivativesTable = self:calculateCostFunctionDerivatives(self.learningRate, deltaTable, numberOfData)
	
	if not doNotUpdateModelParameters then
		
		self.ModelParameters = self:gradientDescent(costFunctionDerivativesTable)
		
	end

	if clearTables then

		self.forwardPropagateTable = nil

		self.zTable = nil

	end

	return costFunctionDerivativesTable

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

	local numberOfFeatures = #featureMatrix[1]
	
	local numberOfNeuronsAtInputLayer = self.numberOfNeuronsTable[1] + self.hasBiasNeuronTable[1]

	if (numberOfNeuronsAtInputLayer ~= numberOfFeatures) then error("Input layer has " .. numberOfNeuronsAtInputLayer .. " neuron(s), but feature matrix has " .. #featureMatrix[1] .. " features!") end

	if (#featureMatrix ~= #labelVector) then error("Number of rows of feature matrix and the label vector is not the same!") end
	
	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]
	
	local numberOfIterations = 0

	local cost

	local costArray = {}

	local deltaTable

	local RegularizationDerivatives

	local lossMatrix

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

			return self:calculateCost(activatedOutputsMatrix, logisticMatrix)

		end)

		if cost then 

			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)

			if (math.abs(cost) <= self.targetCost) then break end

		end

		lossMatrix = AqwamMatrixLibrary:subtract(activatedOutputsMatrix, logisticMatrix)

		self:backPropagate(lossMatrix, true)

	until (numberOfIterations == self.maxNumberOfIterations)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	if (self.AutoResetOptimizers) then

		for i, Optimizer in ipairs(self.OptimizerTable) do

			if Optimizer then Optimizer:reset() end

		end

	end

	return costArray

end

function NeuralNetworkModel:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then self:generateLayers() end

	local outputMatrix = self:forwardPropagate(featureMatrix, false)

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
	local maxRegularizationLength = string.len("Regularization Added")
	
	local hasBias

	for i = 1, #self.numberOfNeuronsTable do

		maxLayerLength = math.max(maxLayerLength, string.len(tostring(i)))

		maxNeuronsLength = math.max(maxNeuronsLength, string.len(tostring(self.numberOfNeuronsTable[i])))
		
		hasBias = (self.hasBiasNeuronTable[i] == 1)
		
		maxBiasLength = math.max(maxBiasLength, string.len(tostring(hasBias)))

		maxActivationLength = math.max(maxActivationLength, string.len(self.activationFunctionTable[i]))

		maxLearningRateLength = math.max(maxLearningRateLength, string.len(tostring(self.learningRateTable[i])))

		maxOptimizerLength = math.max(maxOptimizerLength, string.len("false"))

		maxRegularizationLength = math.max(maxRegularizationLength, string.len("false"))

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
		string.rep("-", maxRegularizationLength) .. "-|" .. 
		"\n"
	
	stringToPrint ..= "| " .. string.format("%-" .. maxLayerLength .. "s", "Layer") .. " | " ..
		string.format("%-" .. maxNeuronsLength .. "s", "Number Of Neurons") .. " | " ..
		string.format("%-" .. maxBiasLength .. "s", "Has Bias Neuron") .. " | " ..
		string.format("%-" .. maxActivationLength .. "s", "Activation Function") .. " | " ..
		string.format("%-" .. maxLearningRateLength .. "s", "Learning Rate") .. " | " ..
		string.format("%-" .. maxOptimizerLength .. "s", "Optimizer Added") .. " | " ..
		string.format("%-" .. maxRegularizationLength .. "s", "Regularization Added") .. " |" .. 
		"\n"
	
	
	stringToPrint ..= "|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxLearningRateLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizationLength) .. "-|" .. 
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

		local regularization = "| " .. string.format("%-" .. maxRegularizationLength .. "s", self.RegularizationTable[i] and "true" or "false") .. " |"
		
		local stringPart = layer .. neurons .. bias .. activation .. learningRate .. optimizer .. regularization .. "\n"
		
		stringToPrint ..= stringPart

	end
	
	stringToPrint ..= "|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxLearningRateLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizationLength) .. "-|".. 
		"\n\n"

	print(stringToPrint)

end

return NeuralNetworkModel
