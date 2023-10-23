local BaseModel = require("Model_BaseModel")

NeuralNetworkModel = {}

NeuralNetworkModel.__index = NeuralNetworkModel

setmetatable(NeuralNetworkModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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

	["None"] = function (zMatrix) return AqwamMatrixLibrary:createMatrix(#zMatrix, #zMatrix[1], 1) end,

}

local cutOffListForScalarValues = {

	["Sigmoid"] = function (a) return (a >= 0.5) end,

	["Tanh"] = function (a) return (a >= 0) end,

	["ReLU"] = function (a) return (a >= 0) end,

	["LeakyReLU"] = function (a) return (a >= 0) end,

	["ELU"] = function (a) return (a >= 0) end,

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

function NeuralNetworkModel:forwardPropagate(featureMatrix)

	local layerZ

	local forwardPropagateTable = {}

	local zTable = {}

	local inputMatrix = featureMatrix

	local numberOfData = #featureMatrix

	local activationFunctionName

	local activationFunction

	local hasBiasNeuron

	local numberOfLayers = #self.ModelParameters

	table.insert(zTable, inputMatrix)

	table.insert(forwardPropagateTable, inputMatrix) -- don't remove this! otherwise the code won't work!

	for layerNumber, weightMatrix in ipairs(self.ModelParameters) do

		activationFunctionName = self.activationFunctionTable[layerNumber]

		activationFunction = activationFunctionList[activationFunctionName]

		layerZ = AqwamMatrixLibrary:dotProduct(inputMatrix, weightMatrix)

		if (typeof(layerZ) == "number") then layerZ = {{layerZ}} end

		inputMatrix = activationFunction(layerZ)

		hasBiasNeuron = self.hasBiasNeuronTable[layerNumber]

		if (layerNumber < numberOfLayers) and (hasBiasNeuron) then

			for data = 1, numberOfData, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron

		end

		table.insert(zTable, layerZ)

		table.insert(forwardPropagateTable, inputMatrix)

	end

	return forwardPropagateTable, zTable

end

function NeuralNetworkModel:backPropagate(lossMatrix, aMatrix, zTable)

	local backpropagateTable = {}

	local numberOfLayers = #self.activationFunctionTable

	local derivativeFunction

	local layerCostMatrix

	local layerMatrix

	local biasMatrix

	local layerMatrixTransposed

	local errorPart1

	local errorPart2

	local errorPart3

	local zLayerMatrix

	local activationFunctionName

	layerCostMatrix = lossMatrix

	table.insert(backpropagateTable, layerCostMatrix)

	for output = #self.ModelParameters, 2, -1 do

		activationFunctionName = self.activationFunctionTable[output]

		derivativeFunction = derivativeList[activationFunctionName]

		layerMatrix = self.ModelParameters[output]

		layerMatrix = AqwamMatrixLibrary:transpose(layerMatrix)

		zLayerMatrix = zTable[output]

		errorPart1 = AqwamMatrixLibrary:dotProduct(layerCostMatrix, layerMatrix)

		errorPart2 = derivativeFunction(aMatrix, zLayerMatrix)

		layerCostMatrix = AqwamMatrixLibrary:multiply(errorPart1, errorPart2)

		table.insert(backpropagateTable, 1, layerCostMatrix)

	end

	return backpropagateTable

end

function NeuralNetworkModel:calculateDelta(forwardPropagateTable, backpropagateTable)

	local partialDerivativeMatrix

	local activationLayerMatrix

	local costFunctionDerivatives

	local deltaTable = {}

	for layer = #backpropagateTable, 1, -1 do

		activationLayerMatrix = forwardPropagateTable[layer]

		partialDerivativeMatrix = AqwamMatrixLibrary:transpose(backpropagateTable[layer])

		costFunctionDerivatives = AqwamMatrixLibrary:dotProduct(partialDerivativeMatrix, activationLayerMatrix)

		costFunctionDerivatives = AqwamMatrixLibrary:transpose(costFunctionDerivatives)

		table.insert(deltaTable, 1, costFunctionDerivatives)

	end

	return deltaTable

end

function NeuralNetworkModel:gradientDescent(learningRate, deltaTable, numberOfData)

	local regularizationDerivatives

	local costFunctionDerivatives

	local newWeightMatrix

	local NewModelParameters = {}

	local calculatedLearningRate = learningRate / numberOfData

	for layerNumber, weightMatrix in ipairs(self.ModelParameters) do

		local costFunctionDerivatives = deltaTable[layerNumber]

		local Regularization = self.RegularizationTable[layerNumber]

		local Optimizer = self.OptimizerTable[layerNumber]

		if Regularization then

			regularizationDerivatives = Regularization:calculateRegularizationDerivatives(self.ModelParameters[layerNumber], numberOfData)

			costFunctionDerivatives = AqwamMatrixLibrary:add(costFunctionDerivatives, costFunctionDerivatives)

		end

		if Optimizer then

			costFunctionDerivatives = Optimizer:calculate(calculatedLearningRate, costFunctionDerivatives)

		else

			costFunctionDerivatives = AqwamMatrixLibrary:multiply(calculatedLearningRate, costFunctionDerivatives)

		end

		newWeightMatrix = AqwamMatrixLibrary:subtract(weightMatrix, costFunctionDerivatives)

		table.insert(NewModelParameters, newWeightMatrix)

	end

	return NewModelParameters

end

function NeuralNetworkModel:calculateCost(allOutputsMatrix, logisticMatrix, numberOfData)

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

	local numberOfLayers = #self.ModelParameters

	local matrixAtFinalLayer = self.ModelParameters[numberOfLayers]

	local numberOfNeuronsAtFinalLayer = #matrixAtFinalLayer[1]

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

		if self.hasBiasNeuronTable[layer] then numberOfCurrentLayerNeurons += 1 end -- 1 is added for bias

		numberOfNextLayerNeurons = layersArray[layer + 1]

		if self.hasBiasNeuronTable[layer + 1] then numberOfNextLayerNeurons += 1 end

		weightMatrix = self:initializeMatrixBasedOnMode(numberOfCurrentLayerNeurons, numberOfNextLayerNeurons)

		table.insert(ModelParameters, weightMatrix)

	end

	self.ModelParameters = ModelParameters

end

function NeuralNetworkModel:createLayers(numberOfNeuronsArray, activationFunction, Optimizer, Regularization)

	activationFunction = activationFunction or defaultActivationFunction

	if (typeof(numberOfNeuronsArray) ~= "table") then error("Invalid input for number of neurons!") end

	if (typeof(activationFunction) ~= "string") then error("Invalid input for activation function!") end

	self.ModelParameters = nil

	self.numberOfNeuronsTable = numberOfNeuronsArray

	self.hasBiasNeuronTable = {}

	self.activationFunctionTable = {}

	self.OptimizerTable = {}

	self.RegularizationTable = {}

	local numberOfLayers = #self.numberOfNeuronsTable

	for layer = 1, numberOfLayers, 1 do

		self.activationFunctionTable[layer] = activationFunction

		if (layer == numberOfLayers) then

			self.hasBiasNeuronTable[layer] = false

			self.OptimizerTable[layer] = Optimizer

			self.RegularizationTable[layer] = Regularization

		else

			self.hasBiasNeuronTable[layer] = true

			self.OptimizerTable[layer] = nil

			self.RegularizationTable[layer] = nil

		end

	end

end

function NeuralNetworkModel:addLayer(numberOfNeurons, hasBiasNeuron, activationFunction, Optimizer, Regularization)

	if (typeof(numberOfNeurons) ~= "number") then error("Invalid input for number of neurons!") end

	local hasBiasNeuronType = typeof(hasBiasNeuron)

	if (hasBiasNeuronType ~= "nil") and (hasBiasNeuronType ~= "boolean") then error("Invalid input for adding bias!") end

	local activationFunctionType = typeof(activationFunction)

	if (activationFunctionType ~= "nil") and (activationFunctionType ~= "string") then error("Invalid input for activation function!") end

	hasBiasNeuron = self:getBooleanOrDefaultOption(hasBiasNeuron, true)

	activationFunction = activationFunction or defaultActivationFunction

	table.insert(self.numberOfNeuronsTable, numberOfNeurons)

	table.insert(self.hasBiasNeuronTable, hasBiasNeuron)

	table.insert(self.activationFunctionTable, activationFunction)

	table.insert(self.OptimizerTable, Optimizer)

	table.insert(self.RegularizationTable, Regularization)

end

function NeuralNetworkModel:setLayer(layerNumber, hasBiasNeuron, activationFunction, Optimizer, Regularization)

	if (typeof(layerNumber) ~= "number") then error("Invalid input layer number!") end

	if (typeof(hasBiasNeuron) ~= "boolean") then error("Invalid input for adding bias!") end

	if  (typeof(activationFunction) ~= "string") then error("Invalid input for activation function!") end 

	self.hasBiasNeuronTable[layerNumber] = hasBiasNeuron or self.hasBiasNeuronTable[layerNumber]

	self.activationFunctionTable[layerNumber] = activationFunction or self.activationFunctionTable[layerNumber] 

	self.OptimizerTable[layerNumber] = Optimizer or self.OptimizerTable[layerNumber]

	self.RegularizationTable[layerNumber] = Regularization or self.RegularizationTable[layerNumber]

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

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector") end

	end

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)

	return logisticMatrix

end

function NeuralNetworkModel:train(featureMatrix, labelVector)

	if (self.ModelParameters == nil) then self:generateLayers() end

	local numberOfFeatures = #featureMatrix[1]

	if (#self.ModelParameters[1] ~= numberOfFeatures) then error("Input layer has " .. #self.ModelParameters[1] .. " neuron(s), but feature matrix has " .. #featureMatrix[1] .. " features!") end

	if (#featureMatrix ~= #labelVector) then error("Number of rows of feature matrix and the label vector is not the same!") end

	local cost

	local costArray = {}

	local numberOfIterations = 0

	local outputMatrix

	local regularizationCost 

	local numberOfData = #featureMatrix

	local numberOfLayers = #self.numberOfNeuronsTable

	local numberOfNeuronsAtFinalLayer = #self.ModelParameters[numberOfLayers - 1][1]

	local transposedLayerMatrix

	local deltaTable

	local RegularizationDerivatives

	local forwardPropagateTable

	local zTable

	local backwardPropagateTable

	local classesList

	local lossMatrix
	
	local aTable

	local logisticMatrix

	local activatedOutputsMatrix

	local finalActivationFunction

	local finalActivationFunctionDerivatives

	local finalActivationFunctionName

	local outputDerivativeMatrix

	local ModelParameters

	if (#labelVector[1] == 1) and (numberOfNeuronsAtFinalLayer ~= 1) then

		logisticMatrix = self:processLabelVector(labelVector)

	else

		if (#labelVector[1] ~= numberOfNeuronsAtFinalLayer) then error("The number of columns for the label matrix is not equal to number of neurons at final layer!") end

		logisticMatrix = labelVector

	end

	finalActivationFunctionName = self.activationFunctionTable[numberOfLayers]

	if (finalActivationFunctionName == "None") then finalActivationFunctionName = self.activationFunctionTable[numberOfLayers - 1] end

	finalActivationFunction = activationFunctionList[finalActivationFunctionName]

	finalActivationFunctionDerivatives = derivativeList[finalActivationFunctionName]

	repeat

		self:iterationWait()

		forwardPropagateTable, aTable, zTable = self:forwardPropagate(featureMatrix)

		outputMatrix = forwardPropagateTable[#forwardPropagateTable]

		activatedOutputsMatrix = finalActivationFunction(outputMatrix)

		cost = self:calculateCost(activatedOutputsMatrix, logisticMatrix, numberOfData)

		lossMatrix = AqwamMatrixLibrary:subtract(activatedOutputsMatrix, logisticMatrix)

		outputDerivativeMatrix = finalActivationFunctionDerivatives(aTable[#aTable], lossMatrix)

		backwardPropagateTable = self:backPropagate(outputDerivativeMatrix, aTable, zTable)

		deltaTable = self:calculateDelta(forwardPropagateTable, backwardPropagateTable)

		ModelParameters = self:gradientDescent(self.learningRate, deltaTable, numberOfData) -- do not refactor the code where the output is self.ModelParameters. Otherwise it cannot update to new model parameters values!

		numberOfIterations += 1

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	self.ModelParameters = ModelParameters

	if (self.AutoResetOptimizers) then

		for i, Optimizer in ipairs(self.OptimizerTable) do

			if Optimizer then Optimizer:reset() end

		end

	end

	return costArray

end

function NeuralNetworkModel:predict(featureMatrix, returnOriginalOutput)

	if (self.ModelParameters == nil) then self:generateLayers() end

	local forwardPropagateTable = self:forwardPropagate(featureMatrix)

	local outputMatrix = forwardPropagateTable[#forwardPropagateTable]

	local finalActivationFunctionName = self.activationFunctionTable[#self.activationFunctionTable]

	local finalActivationFunction = activationFunctionList[finalActivationFunctionName]

	outputMatrix = finalActivationFunction(outputMatrix)

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
	local maxBiasLength = string.len("Bias Neuron Added")
	local maxActivationLength = string.len("Activation Function")
	local maxOptimizerLength = string.len("Optimizer Added")
	local maxRegularizationLength = string.len("Regularization Added")

	for i = 1, #self.numberOfNeuronsTable do

		maxLayerLength = math.max(maxLayerLength, string.len(tostring(i)))

		maxNeuronsLength = math.max(maxNeuronsLength, string.len(tostring(self.numberOfNeuronsTable[i])))

		maxBiasLength = math.max(maxBiasLength, string.len(tostring(self.addBiasNeuronTable[i])))

		maxActivationLength = math.max(maxActivationLength, string.len(self.activationFunctionTable[i]))

		maxOptimizerLength = math.max(maxOptimizerLength, string.len("false"))

		maxRegularizationLength = math.max(maxRegularizationLength, string.len("false"))

	end

	-- Print the table header
	print("Layer Details: \n")

	print("|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizationLength) .. "-|")

	print("| " .. string.format("%-" .. maxLayerLength .. "s", "Layer") .. " | " ..
		string.format("%-" .. maxNeuronsLength .. "s", "Number Of Neurons") .. " | " ..
		string.format("%-" .. maxBiasLength .. "s", "Bias Neuron Added") .. " | " ..
		string.format("%-" .. maxActivationLength .. "s", "Activation Function") .. " | " ..
		string.format("%-" .. maxOptimizerLength .. "s", "Optimizer Added") .. " | " ..
		string.format("%-" .. maxRegularizationLength .. "s", "Regularization Added") .. " |")

	print("|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizationLength) .. "-|")

	-- Print the layer details
	for i = 1, #self.numberOfNeuronsTable do

		local layer = "| " .. string.format("%-" .. maxLayerLength .. "s", i) .. " "

		local neurons = "| " .. string.format("%-" .. maxNeuronsLength .. "s", self.numberOfNeuronsTable[i]) .. " "

		local bias = "| " .. string.format("%-" .. maxBiasLength .. "s", tostring(self.addBiasNeuronTable[i])) .. " "

		local activation = "| " .. string.format("%-" .. maxActivationLength .. "s", self.activationFunctionTable[i]) .. " "

		local optimizer = "| " .. string.format("%-" .. maxOptimizerLength .. "s", self.OptimizerTable[i] and "true" or "false") .. " "

		local regularization = "| " .. string.format("%-" .. maxRegularizationLength .. "s", self.RegularizationTable[i] and "true" or "false") .. " |"

		print(layer .. neurons .. bias .. activation .. optimizer .. regularization)

	end

	print("|-" .. string.rep("-", maxLayerLength) .. "-|-" ..
		string.rep("-", maxNeuronsLength) .. "-|-" ..
		string.rep("-", maxBiasLength) .. "-|-" ..
		string.rep("-", maxActivationLength) .. "-|-" ..
		string.rep("-", maxOptimizerLength) .. "-|-" ..
		string.rep("-", maxRegularizationLength) .. "-|")

end

return NeuralNetworkModel
