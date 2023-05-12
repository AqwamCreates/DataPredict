local BaseModel = require(script.Parent.BaseModel)

NeuralNetworkModel = {}

NeuralNetworkModel.__index = NeuralNetworkModel

setmetatable(NeuralNetworkModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultActivationFunction = "ReLU"

local defaultTargetCost = 0

local defaultLambda = 0

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

local function getClassesList(labelVector)

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

local function convertLabelVectorToLogisticMatrix(modelParameters, labelVector, classesList)
	
	local logisticMatrix

	local lastLayerNumber = #modelParameters

	local layerMatrix = modelParameters[lastLayerNumber]

	local numberOfNeurons = #layerMatrix[1]
	
	if (numberOfNeurons ~= #classesList) then error("The number of classes are not equal to number of neurons. Please adjust your last layer using setLayers() function.") end
	
	if (typeof(labelVector) == "number") then
		
		labelVector = {{labelVector}}
		
	end

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(#labelVector, numberOfNeurons)

	local label
	
	local labelPosition

	for row = 1, #labelVector, 1 do

		label = labelVector[row][1]
		
		labelPosition = table.find(classesList, label)

		logisticMatrix[row][labelPosition] = 1

	end

	return logisticMatrix

end

local function forwardPropagate(featureMatrix, ModelParameters, activationFunction)
	
	local layerZ

	local forwardPropagateTable = {}
	
	local zTable = {}
	
	local biasMatrix = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1, 1)
	
	local featureMatrixWithBias = AqwamMatrixLibrary:horizontalConcatenate(biasMatrix, featureMatrix)
	
	local inputMatrix = featureMatrixWithBias
	
	table.insert(zTable, inputMatrix)
	
	local numberOfLayers = #ModelParameters
	
	table.insert(forwardPropagateTable, inputMatrix) -- don't remove this! otherwise the code won't work!
	
	for layerNumber, weightMatrix in ipairs(ModelParameters) do
		
		layerZ = AqwamMatrixLibrary:dotProduct(inputMatrix, weightMatrix)
		
		table.insert(zTable, layerZ)
		
		inputMatrix = AqwamMatrixLibrary:applyFunction(activationFunctionList[activationFunction], layerZ)
		
		if (layerNumber < numberOfLayers) then
			
			for data = 1, #featureMatrix, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron
			
		end
		
		table.insert(forwardPropagateTable, inputMatrix)
		
	end
	
	return forwardPropagateTable, zTable

end

local function backPropagate(featureMatrix, ModelParameters, lossMatrix, zTable, activationFunction)

	local backpropagateTable = {}

	local numberOfLayers = #ModelParameters

	local layerCostMatrix

	local layerMatrix
	
	local biasMatrix

	local layerMatrixTransposed

	local errorPart1

	local errorPart2

	local errorPart3
	
	local zLayerMatrix
	
	local derivativeFunction = derivativeList[activationFunction]
	
	layerCostMatrix = lossMatrix

	table.insert(backpropagateTable, layerCostMatrix)

	for output = numberOfLayers, 2, -1 do
		
		layerMatrix = ModelParameters[output]
		
		layerMatrix = AqwamMatrixLibrary:transpose(layerMatrix)
		
		zLayerMatrix = zTable[output]
		
		errorPart1 = AqwamMatrixLibrary:dotProduct(layerCostMatrix, layerMatrix)
		
		errorPart2 = AqwamMatrixLibrary:applyFunction(derivativeFunction, zLayerMatrix)

		layerCostMatrix = AqwamMatrixLibrary:multiply(errorPart1, errorPart2)

		table.insert(backpropagateTable, 1, layerCostMatrix)

	end

	return backpropagateTable

end

local function calculateDelta(forwardPropagateTable, backpropagateTable)
	
	local deltaMatrix
	
	local partialDerivativeMatrix
	
	local activationLayerMatrix
	
	local deltaTable = {}
	
	for layer = #backpropagateTable, 1, -1 do

		activationLayerMatrix = forwardPropagateTable[layer]

		partialDerivativeMatrix = AqwamMatrixLibrary:transpose(backpropagateTable[layer])

		deltaMatrix = AqwamMatrixLibrary:dotProduct(partialDerivativeMatrix, activationLayerMatrix)
		
		deltaMatrix = AqwamMatrixLibrary:transpose(deltaMatrix)

		table.insert(deltaTable, 1, deltaMatrix)
		
	end
	
	return deltaTable
	
end

local function gradientDescent(learningRate, ModelParameters, deltaTable, numberOfData)
	
	local costFunctionDerivative
	
	local newWeightMatrix
	
	local NewModelParameters = {}
	
	local calculatedLearningRate = learningRate / numberOfData
	
	for layerNumber, weightMatrix in ipairs(ModelParameters) do
		
		costFunctionDerivative = AqwamMatrixLibrary:multiply(calculatedLearningRate, deltaTable[layerNumber])
		
		newWeightMatrix = AqwamMatrixLibrary:add(weightMatrix, costFunctionDerivative)

		table.insert(NewModelParameters, newWeightMatrix)
		
	end
	
	return NewModelParameters

end

local function punish(punishValue, ModelParameters, deltaTable)

	local costFunctionDerivative
	
	local newWeightMatrix

	local NewModelParameters = {}

	for layerNumber, weightMatrix in ipairs(ModelParameters) do

		costFunctionDerivative = AqwamMatrixLibrary:multiply(punishValue, deltaTable[layerNumber])

		newWeightMatrix = AqwamMatrixLibrary:subtract(weightMatrix, costFunctionDerivative)

		table.insert(NewModelParameters, newWeightMatrix)

	end

	return NewModelParameters

end

local function calculateErrorVector(allOutputsMatrix, logisticMatrix, numberOfData)

	local subtractedMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local squaredSubtractedMatrix = AqwamMatrixLibrary:power(subtractedMatrix, 2)

	local sumSquaredSubtractedMatrix = AqwamMatrixLibrary:verticalSum(squaredSubtractedMatrix)

	local errorVector = AqwamMatrixLibrary:multiply((1/numberOfData), sumSquaredSubtractedMatrix)

	return errorVector

end

local function getLabelFromOutputVector(outputVector, classesList)
	
	local highestValue = math.max(unpack(outputVector[1]))

	local labelPosition = table.find(outputVector[1], highestValue)
	
	local label = classesList[labelPosition]

	return label

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, classesList)
	
	local labelVectorColumn = AqwamMatrixLibrary:transpose(labelVector)
	
	for i, value in ipairs(labelVectorColumn[1]) do
		
		if table.find(classesList, value) then continue end
		
		return true
		
	end
	
	return false
	
end

local function checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)
	
	if (rewardValue == nil) then error("Reward value is nil!") end

	if (punishValue == nil) then error("Punish value is nil!") end

	if (rewardValue < 0) then error("Reward value must be a positive integer!") end

	if (punishValue < 0) then error("Punish value must be a positive integer!") end
	
end


function NeuralNetworkModel.new(maxNumberOfIterations, learningRate, activationFunction, targetCost)

	local NewNeuralNetworkModel = BaseModel.new()

	setmetatable(NewNeuralNetworkModel, NeuralNetworkModel)

	NewNeuralNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewNeuralNetworkModel.learningRate = learningRate or defaultLearningRate

	NewNeuralNetworkModel.activationFunction = activationFunction or defaultActivationFunction

	NewNeuralNetworkModel.targetCost = targetCost or defaultTargetCost

	NewNeuralNetworkModel.validationFeatureMatrix = nil

	NewNeuralNetworkModel.validationLabelVector = nil

	NewNeuralNetworkModel.Optimizer = nil
	
	NewNeuralNetworkModel.Regularization = nil
	
	NewNeuralNetworkModel.ClassesList = {}

	return NewNeuralNetworkModel

end

function NeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, activationFunction, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.activationFunction = activationFunction or self.activationFunction

	self.targetCost = targetCost or self.targetCost

end

function NeuralNetworkModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function NeuralNetworkModel:setRegularization(Regularization)

	self.Regularization = Regularization

end

function NeuralNetworkModel:setLayers(...)
	
	local layersArray = {...}
	
	local numberOfLayers = #layersArray
	
	local ModelParameters = {}
	
	local weightMatrix
	
	local numberOfCurrentLayerNeurons
	
	local numberOfNextLayerNeurons
	
	for layer = 1, (numberOfLayers - 2), 1 do
		
		numberOfCurrentLayerNeurons = layersArray[layer] + 1
		
		numberOfNextLayerNeurons = layersArray[layer + 1] + 1 -- 1 is added for bias
		
		weightMatrix = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfCurrentLayerNeurons, numberOfNextLayerNeurons)
		
		table.insert(ModelParameters, weightMatrix)
		
	end
	
	weightMatrix = AqwamMatrixLibrary:createRandomNormalMatrix(layersArray[numberOfLayers - 1] + 1, layersArray[numberOfLayers]) -- final layer, no bias needed.
	
	table.insert(ModelParameters, weightMatrix)
	
	self.ModelParameters = ModelParameters
	
end

function NeuralNetworkModel:train(featureMatrix, labelVector)
	
	if (self.ModelParameters == nil) then error("Layers are not set!")
		
	elseif (#self.ModelParameters[1] ~= (#featureMatrix[1] + 1)) then error("Input layer has " .. (#self.ModelParameters[1] - 1) .. " neuron(s) without the bias, but feature matrix has " .. #featureMatrix[1] .. " features!")
	
	elseif (#featureMatrix ~= #labelVector) then error("Number of rows of feature matrix and the label vector is not the same!") end
	
	local allOutputsMatrix
	
	local costMatrix
	
	local costVector
	
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
	
	local costDerivativeTable
	
	local classesList
	
	local previousDeltaTable
	
	local lossMatrix
	
	if (#self.ClassesList == 0) then
		
		classesList = getClassesList(labelVector)
		
		table.sort(classesList, function(a,b) return a < b end)
		
		self.ClassesList = classesList
		
	else
		
		if checkIfAnyLabelVectorIsNotRecognized(labelVector, classesList) then error("A value does not exist in the neural network\'s classes list is present in the label vector") end
		
	end
	
	local logisticMatrix = convertLabelVectorToLogisticMatrix(self.ModelParameters, labelVector, classesList)
	
	repeat
		
		numberOfIterations += 1
		
		forwardPropagateTable, zTable = forwardPropagate(featureMatrix, self.ModelParameters, self.activationFunction)
		
		allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]
		
		lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix) 
		
		backwardPropagateTable = backPropagate(featureMatrix, self.ModelParameters, lossMatrix, zTable, self.activationFunction)
		
		deltaTable = calculateDelta(forwardPropagateTable, backwardPropagateTable)
		
		if (self.Regularization) then
				
			RegularizationDerivatives = self.Regularization:calculateLossFunctionDerivativeRegularizaion(self.ModelParameters[numberOfLayers], numberOfData)

			deltaTable[numberOfLayers]  = AqwamMatrixLibrary:add(deltaTable[numberOfLayers], RegularizationDerivatives)

		end

		if (self.Optimizer) then 
				
			deltaTable[numberOfLayers] = self.Optimizer:calculate(deltaTable[numberOfLayers], previousDeltaTable[numberOfLayers]) 

		end
		
		self.ModelParameters = gradientDescent(self.learningRate, self.ModelParameters, deltaTable, numberOfData) -- do not refactor the code where the output is self.ModelParameters. Otherwise it cannot update to new model parameters values!
		
		costVector = calculateErrorVector(allOutputsMatrix, logisticMatrix, numberOfData)
		
		cost = AqwamMatrixLibrary:sum(costVector)
		
		if (self.Regularization) then
				
			regularizationCost = self.Regularization:calculateLossFunctionRegularization(backwardPropagateTable[numberOfLayers], numberOfData)

			cost += regularizationCost

		end
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
		previousDeltaTable = deltaTable

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	if self.Optimizer then self.Optimizer:reset() end

	return costArray
	
end

function NeuralNetworkModel:predict(featureMatrix)

	local forwardPropagateTable = forwardPropagate(featureMatrix, self.ModelParameters, self.activationFunction)
	
	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]
	
	local label = getLabelFromOutputVector(allOutputsMatrix, self.ClassesList)
	
	return label

end

function NeuralNetworkModel:reinforce(featureVector, label, rewardValue, punishValue)
	
	checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)
	
	local logisticMatrix = convertLabelVectorToLogisticMatrix(self.ModelParameters, label, self.ClassesList)
	
	local forwardPropagateTable, zTable = forwardPropagate(featureVector, self.ModelParameters, self.activationFunction)
	
	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]
	
	local lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)
	
	local backwardPropagateTable = backPropagate(featureVector, self.ModelParameters, lossMatrix, zTable, self.activationFunction)
	
	local deltaTable = calculateDelta(forwardPropagateTable, backwardPropagateTable)
	
	local predictedLabel = getLabelFromOutputVector(allOutputsMatrix, self.ClassesList)
	
	if (predictedLabel == label) then
		
		self.ModelParameters = gradientDescent(rewardValue, self.ModelParameters, deltaTable, 1)
		
	else
		
		self.ModelParameters = punish(punishValue, self.ModelParameters, deltaTable)
		
	end
	
	return predictedLabel
	
end

function NeuralNetworkModel:getClassesList()
	
	return self.ClassesList
	
end

function NeuralNetworkModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function NeuralNetworkModel:startQueuedReinforcement(rewardValue, punishValue, showPredictedLabel, showIdleWarning)
	
	if (self.IsQueuedReinforcementRunning == true) then error("Queued reinforcement is already active!") end
	
	checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)
	
	self.FeatureVectorQueue = {}
	
	self.LabelQueue = {}
	
	self.IsQueuedReinforcementRunning = true
	
	self.WaitDuration = 0
	
	self.WarningIssued = false
	
	if (showIdleWarning == nil) then showIdleWarning = true else showIdleWarning = showIdleWarning end
	
	if (showPredictedLabel == nil) then showPredictedLabel = false else showPredictedLabel = showPredictedLabel end
	
	local predictedLabel
	
	local queuedReinforceCoroutine = coroutine.resume(coroutine.create(function()
		
		repeat
			
			if (#self.FeatureVectorQueue == 0) or (#self.LabelQueue == 0) then 
				
				task.wait(0.1)
				
				self.WaitDuration += 0.1
				
			elseif (self.IsQueuedReinforcementRunning == false) then 
				
				break
				
			else
				
				predictedLabel = self:reinforce(self.FeatureVectorQueue[1], self.LabelQueue[1], rewardValue, punishValue)
				
				if (showPredictedLabel == true) then print("Predicted Label: " .. predictedLabel .. "\t\t\tActual Label: " .. self.LabelQueue[1]) end

				table.remove(self.FeatureVectorQueue, 1)

				table.remove(self.LabelQueue, 1)
				
				self.WaitDuration = 0
				
				self.WarningIssued = false
				
			end
			
			if (self.WaitDuration >= 30) and (self.WarningIssued == false) and (showIdleWarning == true) then 
				
				warn("The neural network has been idle for more than 30 seconds. Leaving it idle may use unnecessary resource.") 
				
				self.WarningIssued = true	
				
			end
			
		until (self.IsQueuedReinforcementRunning == false)
		
		self.FeatureVectorQueue = nil

		self.LabelQueue = nil

		self.IsQueuedReinforcementRunning = nil
		
		self.WaitDuration = nil
		
		self.WarningIssued = nil	
		
		showPredictedLabel = nil
		
		showIdleWarning = nil
		
		predictedLabel = nil
		
	end))
	
	return queuedReinforceCoroutine
	
end

function NeuralNetworkModel:stopQueuedReinforcement()
	
	self.IsQueuedReinforcementRunning = false
	
end

function NeuralNetworkModel:addFeatureVectorToReinforcementQueue(featureVector)
	
	if (self.IsQueuedReinforcementRunning == nil) then error("Queued reinforcement is not active!") end
	
	table.insert(self.FeatureVectorQueue, featureVector)
	
end

function NeuralNetworkModel:addLabelToReinforcementQueue(label)
	
	if (self.IsQueuedReinforcementRunning == nil) then error("Queued reinforcement is not active!") end
	
	table.insert(self.LabelQueue, label)

end

return NeuralNetworkModel
