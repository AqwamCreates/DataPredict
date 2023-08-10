local BaseModel = require(script.Parent.BaseModel)

NeuralNetworkModel = {}

NeuralNetworkModel.__index = NeuralNetworkModel

setmetatable(NeuralNetworkModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultActivationFunction = "ReLU"

local defaultTargetCost = 0

local activationFunctionList = {

	["sigmoid"] = function (z) return 1/(1+math.exp(-1 * z)) end,

	["tanh"] = function (z) return math.tanh(z) end,

	["ReLU"] = function (z) return math.max(0, z) end,

	["LeakyReLU"] = function (z) return math.max((0.01 * z), z) end,

	["ELU"] = function (z) return if (z > 0) then z else (0.01 * (math.exp(z) - 1)) end

}

local derivativeList = {

	["sigmoid"] = function (z) 

		local a = activationFunctionList["sigmoid"](z)

		return (a * (1-a))

	end,

	["tanh"] = function (z)

		local a = activationFunctionList["tanh"](z)

		return (1 - math.pow(a, 2))

	end,

	["ReLU"] = function (z)

		if (z > 0) then return 1

		else return 0 end

	end,

	["LeakyReLU"] = function (z)

		if (z > 0) then return 1

		else return 0.01 end

	end,

	["ELU"] = function (z)

		if (z > 0) then return 1

		else return 0.01 * math.exp(z) end

	end,

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

function NeuralNetworkModel:convertLabelVectorToLogisticMatrix(labelVector)
	
	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	if (numberOfNeuronsAtFinalLayer ~= #self.ClassesList) then error("The number of classes are not equal to number of neurons. Please adjust your last layer using setLayers() function.") end

	if (typeof(labelVector) == "number") then

		labelVector = {{labelVector}}

	end

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(#labelVector, numberOfNeuronsAtFinalLayer)

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

	table.insert(zTable, inputMatrix)

	local numberOfLayers = #self.ModelParameters

	table.insert(forwardPropagateTable, inputMatrix) -- don't remove this! otherwise the code won't work!

	for layerNumber, weightMatrix in ipairs(self.ModelParameters) do

		layerZ = AqwamMatrixLibrary:dotProduct(inputMatrix, weightMatrix)

		table.insert(zTable, layerZ)

		inputMatrix = AqwamMatrixLibrary:applyFunction(activationFunctionList[self.activationFunctionTable[layerNumber]], layerZ)

		if (layerNumber < numberOfLayers) and (self.hasBiasNeuronTable[layerNumber]) then

			for data = 1, #featureMatrix, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron

		end

		table.insert(forwardPropagateTable, inputMatrix)

	end

	return forwardPropagateTable, zTable

end

function NeuralNetworkModel:backPropagate(lossMatrix, zTable)

	local backpropagateTable = {}

	local numberOfLayers = #self.ModelParameters

	local derivativeFunction

	local layerCostMatrix

	local layerMatrix

	local biasMatrix

	local layerMatrixTransposed

	local errorPart1

	local errorPart2

	local errorPart3

	local zLayerMatrix

	layerCostMatrix = lossMatrix

	table.insert(backpropagateTable, layerCostMatrix)

	for output = numberOfLayers, 2, -1 do

		derivativeFunction = derivativeList[self.activationFunctionTable[output]]

		layerMatrix = self.ModelParameters[output]

		layerMatrix = AqwamMatrixLibrary:transpose(layerMatrix)

		zLayerMatrix = zTable[output]

		errorPart1 = AqwamMatrixLibrary:dotProduct(layerCostMatrix, layerMatrix)

		errorPart2 = AqwamMatrixLibrary:applyFunction(derivativeFunction, zLayerMatrix)

		layerCostMatrix = AqwamMatrixLibrary:multiply(errorPart1, errorPart2)

		table.insert(backpropagateTable, 1, layerCostMatrix)

	end

	return backpropagateTable

end

function NeuralNetworkModel:calculateDelta(forwardPropagateTable, backpropagateTable, numberOfData)

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
		
		if self.OptimizerTable[layerNumber] then

			costFunctionDerivatives = self.OptimizerTable[layerNumber]:calculate(calculatedLearningRate, costFunctionDerivatives)
			
		else
			
			costFunctionDerivatives = AqwamMatrixLibrary:multiply(calculatedLearningRate, costFunctionDerivatives)

		end

		if self.RegularizationTable[layerNumber] then

			regularizationDerivatives = self.RegularizationTable[layerNumber]:calculateRegularizationDerivatives(self.ModelParameters[layerNumber], numberOfData)

			costFunctionDerivatives = AqwamMatrixLibrary:add(costFunctionDerivatives, costFunctionDerivatives)

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

function NeuralNetworkModel:getLabelFromOutputVector(outputVector)
	
	local z = AqwamMatrixLibrary:applyFunction(math.exp, outputVector)
	
	local zSum = AqwamMatrixLibrary:sum(z)
	
	local softmax = AqwamMatrixLibrary:divide(z, zSum)
	
	local highestProbability, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(softmax)
	
	local label
	
	if classIndex then
		
		label = self.ClassesList[classIndex[2]]
		
	end

	return label, highestProbability

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, classesList)

	local labelVectorColumn = AqwamMatrixLibrary:transpose(labelVector)

	for i, value in ipairs(labelVectorColumn[1]) do

		if table.find(classesList, value) then continue end

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

	if (typeof(hasBiasNeuron) ~= "boolean") then error("Invalid input for adding bias!") end

	if (typeof(activationFunction) ~= "string") then error("Invalid input for activation function!") end

	table.insert(self.numberOfNeuronsTable, numberOfNeurons)

	table.insert(self.hasBiasNeuronTable, hasBiasNeuron)

	table.insert(self.activationFunctionTable, activationFunction)

	table.insert(self.OptimizerTable, Optimizer)

	table.insert(self.RegularizationTable, Regularization)

end

function NeuralNetworkModel:setLayer(layerNumber, hasBiasNeuron, activationFunction, Optimizer, Regularization)

	if (typeof(layerNumber) ~= "number") then error("Invalid input layer number!") end

	if (typeof(hasBiasNeuron) ~= "boolean") then error("Invalid input for adding bias!") end

	if (typeof(activationFunction) ~= "string") then error("Invalid input for activation function!") end

	self.hasBiasNeuronTable[layerNumber] = hasBiasNeuron or self.hasBiasNeuronTable[layerNumber]
	
	self.activationFunctionTable[layerNumber] = activationFunction or self.activationFunctionTable[layerNumber] 

	self.OptimizerTable[layerNumber] = Optimizer or self.OptimizerTable[layerNumber]

	self.RegularizationTable[layerNumber] = Regularization or self.RegularizationTable[layerNumber]

end

function NeuralNetworkModel:processLabelVector(labelVector)
	
	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		table.sort(self.ClassesList, function(a,b) return a < b end)

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

	local allOutputsMatrix

	local cost

	local costArray = {}

	local numberOfIterations = 0

	local allOutputsMatrix

	local regularizationCost 

	local numberOfData = #featureMatrix

	local numberOfLayers = #self.ModelParameters

	local transposedLayerMatrix

	local deltaTable

	local RegularizationDerivatives

	local forwardPropagateTable

	local zTable

	local backwardPropagateTable

	local classesList

	local lossMatrix
	
	local logisticMatrix
	
	if (#labelVector[1] == 1) then
		
		logisticMatrix = self:processLabelVector(labelVector)
		
	else
		
		logisticMatrix = labelVector
		
	end

	repeat
		
		self:iterationWait()

		numberOfIterations += 1

		forwardPropagateTable, zTable = self:forwardPropagate(featureMatrix)

		allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]

		lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix) 

		backwardPropagateTable = self:backPropagate(lossMatrix, zTable)

		deltaTable = self:calculateDelta(forwardPropagateTable, backwardPropagateTable, numberOfData)

		self.ModelParameters = self:gradientDescent(self.learningRate, deltaTable, numberOfData) -- do not refactor the code where the output is self.ModelParameters. Otherwise it cannot update to new model parameters values!

		cost = self:calculateCost(allOutputsMatrix, logisticMatrix, numberOfData)

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	if (self.AutoResetOptimizers) then
		
		for i, Optimizer in ipairs(self.OptimizerTable) do

			if Optimizer then Optimizer:reset() end

		end
		
	end
	
	return costArray

end

function NeuralNetworkModel:predict(featureMatrix, returnOriginalOutput)

	local forwardPropagateTable = self:forwardPropagate(featureMatrix)

	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]
	
	if (returnOriginalOutput == true) then return allOutputsMatrix end

	local label, highestProbability = self:getLabelFromOutputVector(allOutputsMatrix)

	return label, highestProbability

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
