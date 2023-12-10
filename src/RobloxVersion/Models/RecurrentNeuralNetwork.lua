local BaseModel = require(script.Parent.BaseModel)

RecurrentNeuralNetworkModel = {}

RecurrentNeuralNetworkModel.__index = RecurrentNeuralNetworkModel

setmetatable(RecurrentNeuralNetworkModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.001

local defaultActivationFunction = "tanh"

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

local function softMax(matrix)

	local e = AqwamMatrixLibrary:applyFunction(math.exp, matrix)

	local eSum = AqwamMatrixLibrary:sum(e)

	local result = AqwamMatrixLibrary:divide(e, eSum)

	return result

end

function RecurrentNeuralNetworkModel.new(maxNumberOfIterations, learningRate, activationFunction, targetCost)

	local NewRecurrentNeuralNetworkModel = BaseModel.new()

	setmetatable(NewRecurrentNeuralNetworkModel, RecurrentNeuralNetworkModel)

	NewRecurrentNeuralNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewRecurrentNeuralNetworkModel.learningRate = learningRate or defaultLearningRate

	NewRecurrentNeuralNetworkModel.activationFunction = activationFunction or defaultActivationFunction

	NewRecurrentNeuralNetworkModel.targetCost = targetCost or defaultTargetCost

	return NewRecurrentNeuralNetworkModel

end

function RecurrentNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, activationFunction, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.activationFunction = activationFunction or self.activationFunction

	self.targetCost = targetCost or self.targetCost

end

function RecurrentNeuralNetworkModel:createLayers(inputSize, hiddenSize, outputSize)

	self.inputSize = inputSize or self.inputSize

	self.hiddenSize = hiddenSize or self.hiddenSize

	self.outputSize = outputSize or self.outputSize

	if (inputSize == nil) and (hiddenSize == nil) and (outputSize == nil) then return nil end

	self.ModelParameters = nil

end

function RecurrentNeuralNetworkModel:convertTokenToLogisticVector(token)
	
	if (type(token) == nil) then error("A token is not an integer!") end

	if (token > self.outputSize) then error("A token value is larger than the number of output neurons!") end

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)

	if (token ~= 0) then

		logisticMatrix[token][1] = 1

	end

	return logisticMatrix

end

function RecurrentNeuralNetworkModel:forwardPropagateCell(xt, aPrevious)

	local zNextPart1 = AqwamMatrixLibrary:dotProduct(self.Wax, xt)

	local zNextPart2 = AqwamMatrixLibrary:dotProduct(self.Waa, aPrevious)

	local zNext = AqwamMatrixLibrary:add(zNextPart1, zNextPart2, self.ba)

	local activationFunction = activationFunctionList[self.activationFunction]

	local aNext =  AqwamMatrixLibrary:applyFunction(activationFunction, zNext)

	return aNext

end

function RecurrentNeuralNetworkModel:calculatePrediction(aNext)

	local ytPredictionPart1 = AqwamMatrixLibrary:dotProduct(self.Wya, aNext)

	local ytPredictionPart2 = AqwamMatrixLibrary:add(ytPredictionPart1, self.by)

	local ytPrediction = softMax(ytPredictionPart2)

	return ytPrediction

end

function RecurrentNeuralNetworkModel:backwardPropagateCell(daNext, aNext, aPrevious, xt)

	local xtTransposed = AqwamMatrixLibrary:transpose(xt)

	local WaxTransposed = AqwamMatrixLibrary:transpose(self.Wax)

	local aPreviousTransposed = AqwamMatrixLibrary:transpose(aPrevious)

	local WaaTransposed = AqwamMatrixLibrary:transpose(self.Waa)

	local derivativeFunction = derivativeList[self.activationFunction]

	local derivativePart1 = AqwamMatrixLibrary:applyFunction(derivativeFunction, aNext)

	local da = AqwamMatrixLibrary:multiply(derivativePart1, daNext)

	local dWax = AqwamMatrixLibrary:dotProduct(da, xtTransposed)

	local dxt = AqwamMatrixLibrary:dotProduct(WaxTransposed, da)

	local dWaa = AqwamMatrixLibrary:dotProduct(da, aPreviousTransposed)

	local daPrevious = AqwamMatrixLibrary:dotProduct(WaaTransposed, da)

	local dba = AqwamMatrixLibrary:sum(da)

	return dxt, daPrevious, dWax, dWaa, dba

end

function RecurrentNeuralNetworkModel:loadModelParameters()

	self.Wax = self.ModelParameters[1]

	self.Waa = self.ModelParameters[2]

	self.Wya = self.ModelParameters[3]

	self.ba = self.ModelParameters[4]

	self.by = self.ModelParameters[5]

end

function RecurrentNeuralNetworkModel:setOptimizers(InputLayerOptimizer, HiddenLayerOptimizer, OutputLayerOptimizer, BiasHiddenLayerOptimizer, BiasOutputLayerOptimizer)

	self.InputLayerOptimizer = InputLayerOptimizer

	self.HiddenLayerOptimizer = HiddenLayerOptimizer

	self.OutputLayerOptimizer = OutputLayerOptimizer

	self.BiasHiddenLayerOptimizer = BiasHiddenLayerOptimizer

	self.BiasOutputLayerOptimizer = BiasOutputLayerOptimizer

end

function RecurrentNeuralNetworkModel:setRegularizations(InputLayerRegularization, HiddenLayerRegularization, OutputLayerRegularization, BiasHiddenLayerRegularization, BiasOutputLayerRegularization)

	self.InputLayerRegularization = InputLayerRegularization

	self.HiddenLayerRegularization = HiddenLayerRegularization

	self.OutputLayerRegularization = OutputLayerRegularization

	self.BiasHiddenLayerRegularization = BiasHiddenLayerRegularization

	self.BiasOutputLayerRegularization = BiasOutputLayerRegularization

end

local function throwErrorIfSequenceLengthAreNotEqual(tokenInputSequenceArray, tokenOutputSequenceArray)
	
	if (tokenOutputSequenceArray == nil) then return nil end
		
	local tokenInputSequenceLength = #tokenInputSequenceArray
		
	local tokenOutputSequenceLength = #tokenOutputSequenceArray

	if (tokenInputSequenceLength ~= tokenOutputSequenceLength) then error("The length of token input and output sequence arrays are not equal!") end
	
end

function RecurrentNeuralNetworkModel:train(tableOfTokenInputSequenceArray, tableOfTokenOutputSequenceArray)

	if (self.ModelParameters) then

		self:loadModelParameters()

	elseif (self.inputSize == nil) or (self.hiddenSize == nil) or (self.outputSize == nil) then

		error("Layers are not set!")

	else

		self.Wax = self:initializeMatrixBasedOnMode(self.hiddenSize, self.inputSize)

		self.Waa = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize)

		self.Wya = self:initializeMatrixBasedOnMode(self.outputSize, self.hiddenSize)

		self.ba = self:initializeMatrixBasedOnMode(self.hiddenSize, 1)

		self.by = self:initializeMatrixBasedOnMode(self.outputSize, 1)

	end
	
	local totalNumberOfTokens = 0

	local numberOfIterations = 0

	local costArray = {}
	
	local tableOfTokenInputSequenceLogisticMatrices = {}
	
	local tableOfTokenOutputSequenceLogisticMatrices = {}
	
	if (tableOfTokenOutputSequenceArray == nil) then tableOfTokenOutputSequenceArray = tableOfTokenInputSequenceArray end

	for s = 1, #tableOfTokenInputSequenceArray, 1 do
		
		throwErrorIfSequenceLengthAreNotEqual(tableOfTokenInputSequenceArray[s], tableOfTokenOutputSequenceArray[s])

		local tokenInputSequenceLogisticMatrices = {}

		local tokenOutputSequenceLogisticMatrices = {}

		for t = 1, #tableOfTokenInputSequenceArray[s], 1 do

			local tokenInput = tableOfTokenInputSequenceArray[s][t]

			local tokenOutput = tableOfTokenOutputSequenceArray[s][t]

			local xt = self:convertTokenToLogisticVector(tokenInput)

			local yt = self:convertTokenToLogisticVector(self.outputSize, tokenInput)

			table.insert(tokenInputSequenceLogisticMatrices, xt)

			table.insert(tokenOutputSequenceLogisticMatrices, yt)

			totalNumberOfTokens += 1

		end

		table.insert(tableOfTokenInputSequenceLogisticMatrices, tokenInputSequenceLogisticMatrices)

		table.insert(tableOfTokenOutputSequenceLogisticMatrices, tokenOutputSequenceLogisticMatrices)

	end

	repeat
		
		self:iterationWait()

		numberOfIterations += 1

		local cost = 0

		local partialCost = 0

		local dWax = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.inputSize)

		local dWaa = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize)

		local dWya = AqwamMatrixLibrary:createMatrix(self.outputSize, self.hiddenSize)

		local dba = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local dby = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)
		
		for s = 1, #tableOfTokenInputSequenceArray, 1 do
			
			self:dataWait()
			
			local xTable = tableOfTokenInputSequenceLogisticMatrices[s]
			
			local yTable = tableOfTokenOutputSequenceLogisticMatrices[s]
			
			local aTable = {}

			local ytPredictionTable = {}

			local daTable = {}

			local tokenInput

			local xt
			
			local yt 

			local aFirst = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)

			local aPrevious = aFirst

			local aNext

			local ytPrediction

			local daNext

			local dxt

			local daPrevious

			local dWaxt

			local dWaat

			local dWyat

			local dbat

			local dat

			for t = 1, #xTable, 1 do
				
				self:sequenceWait()

				xt = xTable[t]

				aNext = self:forwardPropagateCell(xt, aPrevious)

				ytPrediction = self:calculatePrediction(aNext)

				dat = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

				aPrevious = aNext

				table.insert(aTable, aNext)

				table.insert(ytPredictionTable, ytPrediction)

				table.insert(daTable, dat)

			end

			for t = #xTable, 1, -1 do
				
				self:sequenceWait()

				if (t > 1) then

					aPrevious = aTable[t-1]

				else

					aPrevious = aFirst

				end

				ytPrediction = ytPredictionTable[t]

				xt = xTable[t]
				
				yt = yTable[t]

				aNext = aTable[t]

				daNext = daTable[t]

				dxt, daPrevious, dWaxt, dWaat, dbat = self:backwardPropagateCell(daNext, aNext, aPrevious, xt)

				dWyat = AqwamMatrixLibrary:subtract(ytPrediction, yt)

				if (t > 1) then daTable[t-1] = AqwamMatrixLibrary:add(daNext, daPrevious) end

				dWax = AqwamMatrixLibrary:add(dWax, dWaxt)

				dWaa = AqwamMatrixLibrary:add(dWaa, dWaat)

				dba = AqwamMatrixLibrary:add(dba, dbat)
				
				dby = AqwamMatrixLibrary:add(dby, dxt)

				dWya = AqwamMatrixLibrary:add(dWya, dWyat)

				partialCost = AqwamMatrixLibrary:sum(dWya)

				cost += partialCost

			end
			
		end

		cost = cost / totalNumberOfTokens

		if (self.InputLayerRegularization) then

			local regularizationDerivatives =  self.InputLayerRegularization:calculateRegularizationDerivatives(self.Wax, 1)

			dWax = AqwamMatrixLibrary:add(dWax, regularizationDerivatives)

		end

		if (self.HiddenLayerRegularization) then

			local regularizationDerivatives =  self.HiddenLayerRegularization:calculateRegularizationDerivatives(self.Waa, 1)

			dWaa = AqwamMatrixLibrary:add(dWaa, regularizationDerivatives)

		end

		if (self.OutputLayerRegularization) then

			local regularizationDerivatives =  self.OutputLayerRegularization:calculateRegularizationDerivatives(self.Wya, 1)

			dWya = AqwamMatrixLibrary:add(dWya, regularizationDerivatives)

		end

		if (self.BiasHiddenLayerRegularization) then

			local regularizationDerivatives =  self.BiasHiddenLayerRegularization:calculateRegularizationDerivatives(self.ba, 1)

			dba = AqwamMatrixLibrary:add(dba, regularizationDerivatives)

		end

		if (self.BiasOutputLayerRegularization) then

			local regularizationDerivatives =  self.BiasOutputLayerRegularization:calculateRegularizationDerivatives(self.by, 1)

			dby = AqwamMatrixLibrary:add(dby, regularizationDerivatives)

		end
		
		--------------------------------------------------------------------------------------------------------

		if (self.InputLayerOptimizer) then

			dWax = self.InputLayerOptimizer:calculate(self.learningRate, dWax)
			
		else
			
			dWax = AqwamMatrixLibrary:multiply(self.learningRate, dWax)

		end

		if (self.HiddenLayerOptimizer) then

			dWaa = self.HiddenLayerOptimizer:calculate(self.learningRate, dWaa)
			
		else
			
			dWaa = AqwamMatrixLibrary:multiply(self.learningRate, dWaa)

		end

		if (self.OutputLayerOptimizer) then

			dWya = self.OutputLayerOptimizer:calculate(self.learningRate, dWya)
			
		else
			
			dWya = AqwamMatrixLibrary:multiply(self.learningRate, dWya)

		end

		if (self.BiasHiddenLayerOptimizer) then

			dba = self.BiasHiddenLayerOptimizer:calculate(self.learningRate, dba)
			
		else
			
			dba = AqwamMatrixLibrary:multiply(self.learningRate, dba)

		end

		if (self.BiasOutputLayerOptimizer) then

			dby = self.BiasOutputLayerOptimizer:calculate(self.learningRate, dby)
			
		else
			
			dby = AqwamMatrixLibrary:multiply(self.learningRate, dby)

		end

		self.Wax = AqwamMatrixLibrary:subtract(self.Wax, dWax)

		self.Waa = AqwamMatrixLibrary:subtract(self.Waa, dWaa)

		self.Wya = AqwamMatrixLibrary:subtract(self.Wya, dWya)

		self.ba = AqwamMatrixLibrary:subtract(self.ba, dba)

		self.by = AqwamMatrixLibrary:subtract(self.by, dby)

		self.ModelParameters = {self.Wax, self.Waa, self.Wya, self.ba, self.by}

		cost = math.abs(cost)

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (cost <= self.targetCost)

	if (self.InputLayerOptimizer) then

		self.InputLayerOptimizer:reset()

	end

	if (self.HiddenLayerOptimizer) then

		self.HiddenLayerOptimizer:reset()

	end

	if (self.OutputLayerOptimizer) then

		self.OutputLayerOptimizer:reset()

	end

	if (self.BiasHiddenLayerOptimizer) then

		self.BiasHiddenLayerOptimizer:reset()

	end

	if (self.BiasOutputLayerOptimizer) then

		self.BiasOutputLayerOptimizer:reset()

	end

	return costArray

end

function RecurrentNeuralNetworkModel:predict(tableOfTokenInputSequenceLogisticMatrices, returnOriginalOutput)

	if (self.ModelParameters == nil) then error("No Model Parameters Found!") end

	self:loadModelParameters()
	
	local tableOfTokenOutputSequenceArray = {}
	
	for i = 1, #tableOfTokenInputSequenceLogisticMatrices, 1 do
		
		local tokenInputSequenceArray = tableOfTokenInputSequenceLogisticMatrices[i]
		
		local aPrevious = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local predictionArray = {}
		
		for j = 1, #tokenInputSequenceArray, 1 do

			local tokenInput = tokenInputSequenceArray[j]

			local xt = self:convertTokenToLogisticVector(self.inputSize, tokenInput)

			local aNext = self:forwardPropagateCell(xt, aPrevious)

			local ytPrediction = self:calculatePrediction(aNext)
			
			if (returnOriginalOutput) then
				
				table.insert(predictionArray, ytPrediction)
				
			else
				
				local _, predictedTokenIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(ytPrediction)

				local predictedToken = 0

				if predictedTokenIndex then predictedToken = predictedTokenIndex[1] end

				predictedToken = predictedToken or 0

				table.insert(predictionArray, predictedToken)
				
			end

			aPrevious = aNext

		end
		
		table.insert(tableOfTokenOutputSequenceArray, predictionArray)
		
	end

	return tableOfTokenOutputSequenceArray

end

return RecurrentNeuralNetworkModel
