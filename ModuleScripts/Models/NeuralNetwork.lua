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

local function convertLabelVectorToLogisticMatrix(modelParameters, labelVector)

	local lastLayerNumber = #modelParameters

	local layerMatrix = modelParameters[lastLayerNumber]

	local numberOfNeurons = #layerMatrix[1]

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(#labelVector, numberOfNeurons)

	local label

	for row = 1, #labelVector, 1 do

		label = labelVector[row][1]

		logisticMatrix[row][label] = 1

	end

	return logisticMatrix

end

local function forwardPropagate(featureMatrix, ModelParameters, activationFunction)
	
	local layerZ

	local forwardPropagateTable = {}
	
	local biasMatrix = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1, 1)
	
	local featureMatrixWithBias = AqwamMatrixLibrary:horizontalConcatenate(biasMatrix, featureMatrix)
	
	local inputMatrix = featureMatrixWithBias
	
	local numberOfLayers = #ModelParameters
	
	for layerNumber, weightMatrix in ipairs(ModelParameters) do
		
		layerZ = AqwamMatrixLibrary:dotProduct(inputMatrix, weightMatrix)
		
		inputMatrix = AqwamMatrixLibrary:applyFunction(activationFunctionList[activationFunction], layerZ)
		
		if (layerNumber < numberOfLayers) then
			
			for data = 1, #featureMatrix, 1 do inputMatrix[data][1] = 1 end -- because we actually calculated the output of previous layers instead of using bias neurons and the model parameters takes into account of bias neuron size, we will set the first column to one so that it remains as bias neuron
			
		end
		
		table.insert(forwardPropagateTable, inputMatrix)
		
	end
	
	return forwardPropagateTable

end

local function backPropagate(featureMatrix, ModelParameters, logisticMatrix, forwardPropagateTable)

	local backpropagateTable = {}

	local numberOfLayers = #ModelParameters

	local layerCostMatrix = AqwamMatrixLibrary:subtract(forwardPropagateTable[numberOfLayers], logisticMatrix)

	table.insert(backpropagateTable, layerCostMatrix)

	local layerMatrix
	
	local biasMatrix

	local layerMatrixTransposed

	local errorPart1

	local errorPart2

	local errorPart3
	
	layerMatrix = ModelParameters[numberOfLayers]
	
	layerMatrix = AqwamMatrixLibrary:transpose(layerMatrix)

	for output = numberOfLayers, 2, -1 do

		errorPart1 = AqwamMatrixLibrary:dotProduct(layerCostMatrix, layerMatrix)

		errorPart2 = AqwamMatrixLibrary:subtract(1, forwardPropagateTable[output - 1])

		errorPart3 = AqwamMatrixLibrary:multiply(forwardPropagateTable[output], errorPart2)

		layerCostMatrix = AqwamMatrixLibrary:multiply(errorPart1, errorPart3)

		table.insert(backpropagateTable, 1, layerCostMatrix)
		
		layerMatrix = ModelParameters[output - 1]
		
		layerMatrix = AqwamMatrixLibrary:transpose(layerMatrix)

	end

	return backpropagateTable

end

local function gradientDescent(learningRate, ModelParameters, backpropagateTable, numberOfData)
	
	local costFunctionDerivativeTable = {}
	
	local calculatedLearningRate = learningRate / numberOfData
	
	for layerNumber, weightMatrix in ipairs(ModelParameters) do
		
		local costFunctionDerivative = AqwamMatrixLibrary:multiply(calculatedLearningRate, backpropagateTable[layerNumber])
		
		local newWeightMatrix = AqwamMatrixLibrary:add(weightMatrix, costFunctionDerivative)
		
		table.insert(costFunctionDerivativeTable, newWeightMatrix)
		
	end
	
	return costFunctionDerivativeTable

end

local function punish(punishValue, ModelParameters, backpropagateTable)

	local costFunctionDerivativeTable = {}

	for layerNumber, weightMatrix in ipairs(ModelParameters) do

		local costFunctionDerivative = AqwamMatrixLibrary:multiply(punishValue, backpropagateTable[layerNumber])

		local newWeightMatrix = AqwamMatrixLibrary:add(weightMatrix, costFunctionDerivative)

		table.insert(costFunctionDerivativeTable, newWeightMatrix)

	end

	return costFunctionDerivativeTable

end

local function calculateErrorVector(allOutputsMatrix, logisticMatrix)

	local subtractedMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local squaredSubtractedMatrix = AqwamMatrixLibrary:power(subtractedMatrix, 2)

	local sumSquaredSubtractedMatrix = AqwamMatrixLibrary:verticalSum(squaredSubtractedMatrix)

	local errorVector = AqwamMatrixLibrary:multiply((1/2), sumSquaredSubtractedMatrix)

	return errorVector

end

local function getLabelFromOutputVector(outputVector)

	local labelsArray = outputVector[1]
	
	local highestValue = math.max(unpack(labelsArray))

	local label = table.find(labelsArray, highestValue)

	return label

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
	
	local biasMatrix
	
	local weightAndBiasMatrix
	
	local numberOfCurrentLayerNeurons
	
	local numberOfNextLayerNeurons
	
	for layer = 1, (numberOfLayers - 2), 1 do
		
		numberOfCurrentLayerNeurons = layersArray[layer]
		
		numberOfNextLayerNeurons = layersArray[layer + 1] + 1 -- 1 is added for bias
		
		biasMatrix = AqwamMatrixLibrary:createMatrix(1, numberOfNextLayerNeurons)
		
		weightMatrix = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfCurrentLayerNeurons, numberOfNextLayerNeurons)
		
		weightAndBiasMatrix = AqwamMatrixLibrary:verticalConcatenate(biasMatrix, weightMatrix)  -- bias layer is added
		
		table.insert(ModelParameters, weightAndBiasMatrix)
		
	end
	
	weightAndBiasMatrix = AqwamMatrixLibrary:createRandomNormalMatrix(layersArray[numberOfLayers - 1] + 1, layersArray[numberOfLayers]) -- final layer, no bias needed.
	
	table.insert(ModelParameters, weightAndBiasMatrix)
	
	self.ModelParameters = ModelParameters
	
end

function NeuralNetworkModel:train(featureMatrix, labelVector)
	
	if (self.ModelParameters == nil) then error("Layers are not set!")
		
	elseif (#self.ModelParameters[1] ~= (#featureMatrix[1] + 1)) then error("Input layer has " .. (#self.ModelParameters[1] - 1) .. " neuron(s) without the bias, but feature matrix has " .. #featureMatrix[1] .. " features!") end
	
	local allOutputsMatrix
	
	local costMatrix
	
	local costVector
	
	local cost
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local logisticMatrix = convertLabelVectorToLogisticMatrix(self.ModelParameters, labelVector)
	
	local allOutputsMatrix
	
	local regularizationCost 
	
	local numberOfData = #featureMatrix
	
	local numberOfLayers = #self.ModelParameters
	
	local transposedLayerMatrix
	
	local delta = {}
	
	local RegularizationDerivatives
	
	local forwardPropagateTable
	
	local backwardPropagateTable
	
	local costDerivativeTable
	
	repeat
		
		numberOfIterations += 1
		
		forwardPropagateTable = forwardPropagate(featureMatrix, self.ModelParameters, self.activationFunction)
		
		allOutputsMatrix = forwardPropagateTable[numberOfLayers]
		
		backwardPropagateTable = backPropagate(featureMatrix, self.ModelParameters, logisticMatrix, forwardPropagateTable)
		
		if (self.Regularization) then
				
			RegularizationDerivatives = self.Regularization:calculateLossFunctionDerivativeRegularizaion(self.ModelParameters[numberOfLayers], numberOfData)

			backwardPropagateTable[numberOfLayers]  = AqwamMatrixLibrary:add(backwardPropagateTable[numberOfLayers], RegularizationDerivatives)

		end

		if (self.Optimizer) then 
				
			backwardPropagateTable[numberOfLayers] = self.Optimizer:calculate(backwardPropagateTable[numberOfLayers], delta) 

		end
		
		costDerivativeTable = gradientDescent(self.learningRate, self.ModelParameters, backwardPropagateTable, numberOfData) -- do not refactor the code where the output is self.ModelParameters. Otherwise it cannot update to new model parameters values!
		
		for layerNumber, weightMatrix in ipairs(self.ModelParameters) do
			
			self.ModelParameters[layerNumber] = AqwamMatrixLibrary:add(self.ModelParameters[layerNumber], costDerivativeTable[layerNumber])
			
		end
		
		delta = AqwamMatrixLibrary:multiply(self.learningRate, backwardPropagateTable[numberOfLayers])
		
		costVector = calculateErrorVector(allOutputsMatrix, logisticMatrix)
		
		cost = AqwamMatrixLibrary:sum(costVector)
		
		if (self.Regularization) then
				
			regularizationCost = self.Regularization:calculateLossFunctionRegularization(backwardPropagateTable[numberOfLayers], numberOfData)

			cost += regularizationCost

		end
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	if self.Optimizer then self.Optimizer:reset() end

	return costArray
	
end

function NeuralNetworkModel:predict(featureMatrix)

	local forwardPropagateTable = forwardPropagate(featureMatrix, self.ModelParameters, self.activationFunction)
	
	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]
	
	local label = getLabelFromOutputVector(allOutputsMatrix)
	
	return label

end

function NeuralNetworkModel:reinforce(featureVector, label, rewardValue, punishValue)
	
	local costDerivativeTable
	
	local forwardPropagateTable = forwardPropagate(featureVector, self.ModelParameters, self.activationFunction)
	
	local allOutputsMatrix = forwardPropagateTable[#self.ModelParameters]
	
	local logisticMatrix = convertLabelVectorToLogisticMatrix(self.ModelParameters, allOutputsMatrix)
	
	local backwardPropagateTable = backPropagate(featureVector, self.ModelParameters, logisticMatrix, forwardPropagateTable)
	
	local predictedLabel = getLabelFromOutputVector(allOutputsMatrix)
	
	if (predictedLabel == label) then
		
		costDerivativeTable = gradientDescent(rewardValue, self.ModelParameters, backwardPropagateTable, 1)
		
	else
		
		costDerivativeTable = punish(punishValue, self.ModelParameters, backwardPropagateTable)
		
	end
	
	for layerNumber, weightMatrix in ipairs(self.ModelParameters) do

		self.ModelParameters[layerNumber] = AqwamMatrixLibrary:add(self.ModelParameters[layerNumber], costDerivativeTable[layerNumber])

	end
	
	return predictedLabel
	
end

return NeuralNetworkModel
