local BaseModel = require(script.Parent.BaseModel)

RecurrentNeuralNetworkModel = {}

RecurrentNeuralNetworkModel.__index = RecurrentNeuralNetworkModel

setmetatable(RecurrentNeuralNetworkModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

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

	local maxValue = AqwamMatrixLibrary:findMaximumValueInMatrix(matrix)

	local subtractedValues = AqwamMatrixLibrary:subtract(matrix, maxValue)

	local p = AqwamMatrixLibrary:applyFunction(math.exp, subtractedValues)

	local sumValues = AqwamMatrixLibrary:sum(p)

	local result

	if (sumValues ~= 0) then

		result = AqwamMatrixLibrary:divide(p, sumValues)

	else

		result = matrix

	end

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

function RecurrentNeuralNetworkModel:convertTokenToLogisticVector(size, token)
	
	if (type(token) == nil) then error("A token is not an integer!") end

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(size, 1)

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

		self.ba = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		self.by = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)

	end

	local previousdWax

	local previousdWaa

	local previousdWya

	local previousdby

	local previousdba
	
	local totalNumberOfTokens = 0

	local numberOfIterations = 0

	local costArray = {}
	
	local tableOfTokenInputSequenceLogisticMatrices = {}
	
	local tableOfTokenOutputSequenceLogisticMatrices = {}
	
	for i, tokenInputSequenceArray in ipairs(tableOfTokenInputSequenceArray) do
		
		local tokenInputSequenceLogisticMatrices = {}
		
		for t = 1, #tokenInputSequenceArray, 1 do

			local tokenInput = tokenInputSequenceArray[t]

			local xt = self:convertTokenToLogisticVector(self.inputSize, tokenInput)

			table.insert(tokenInputSequenceLogisticMatrices, xt)
			
			totalNumberOfTokens += 1

		end
		
		table.insert(tableOfTokenInputSequenceLogisticMatrices, tokenInputSequenceLogisticMatrices)
		
	end

	if (tableOfTokenOutputSequenceArray) then

		for j, tokenOutputSequenceArray in ipairs(tableOfTokenOutputSequenceArray) do
			
			throwErrorIfSequenceLengthAreNotEqual(tableOfTokenInputSequenceArray[j], tokenOutputSequenceArray)

			local tokenOutputSequenceLogisticMatrices = {}

			for t = 1, #tokenOutputSequenceArray, 1 do

				local tokenInput = tokenOutputSequenceArray[t]

				local yt = self:convertTokenToLogisticVector(self.outputSize, tokenInput)

				table.insert(tokenOutputSequenceLogisticMatrices, yt)

			end

			table.insert(tableOfTokenOutputSequenceLogisticMatrices, tokenOutputSequenceLogisticMatrices)

		end

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

				aNext = aTable[t]

				daNext = daTable[t]

				dxt, daPrevious, dWaxt, dWaat, dbat = self:backwardPropagateCell(daNext, aNext, aPrevious, xt)

				if (t > 1) then daTable[t-1] = AqwamMatrixLibrary:add(daNext, daPrevious) end

				dWax = AqwamMatrixLibrary:add(dWax, dWaxt)

				dWaa = AqwamMatrixLibrary:add(dWaa, dWaat)

				dba = AqwamMatrixLibrary:add(dba, dbat)
				
				dby = AqwamMatrixLibrary:add(dby, dxt)

				if (yTable[t]) then

					local yt = yTable[t]

					dWyat = AqwamMatrixLibrary:subtract(ytPrediction, yt)

				else

					dWyat = AqwamMatrixLibrary:subtract(ytPrediction, xt)

				end

				dWya = AqwamMatrixLibrary:add(dWya, dWyat)

				partialCost = AqwamMatrixLibrary:sum(dWya) / self.outputSize

				cost = cost + partialCost

			end
			
		end

		cost = cost / totalNumberOfTokens

		dWax = AqwamMatrixLibrary:multiply(self.learningRate, dWax)

		dWaa = AqwamMatrixLibrary:multiply(self.learningRate, dWaa)

		dWya = AqwamMatrixLibrary:multiply(self.learningRate, dWya)

		dba = AqwamMatrixLibrary:multiply(self.learningRate, dba)
		
		dby = AqwamMatrixLibrary:multiply(self.learningRate, dby)

		if (self.InputLayerOptimizer) then

			dWax = self.InputLayerOptimizer:calculate(dWax, previousdWax)

		end

		if (self.HiddenLayerOptimizer) then

			dWaa = self.HiddenLayerOptimizer:calculate(dWaa, previousdWaa)

		end

		if (self.OutputLayerOptimizer) then

			dWya = self.OutputLayerOptimizer:calculate(dWya, previousdWya)

		end

		if (self.BiasHiddenLayerOptimizer) then

			dba = self.BiasHiddenLayerOptimizer:calculate(dba, previousdba)

		end

		if (self.BiasOutputLayerOptimizer) then

			dby = self.BiasOutputLayerOptimizer:calculate(dby, previousdby)

		end

		previousdWax = dWax

		previousdWaa = dWaa

		previousdWya = dWya

		previousdba = dba

		previousdby = dby

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

function RecurrentNeuralNetworkModel:predict(tokenInputSequenceArray)

	if (self.ModelParameters == nil) then error("No Model Parameters Found!") end

	self:loadModelParameters()

	local aPrevious = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)

	local predictionArray = {}

	for i = 1, #tokenInputSequenceArray, 1 do

		local tokenInput = tokenInputSequenceArray[i]

		local xt = self:convertTokenToLogisticVector(self.inputSize, tokenInput)

		local aNext = self:forwardPropagateCell(xt, aPrevious)

		local ytPrediction = self:calculatePrediction(aNext)

		local _, predictedTokenIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(ytPrediction)

		local predictedToken = nil
		
		if predictedTokenIndex then predictedToken = predictedTokenIndex[1] end

		table.insert(predictionArray, predictedToken)

		aPrevious = aNext

	end

	return predictionArray

end

return RecurrentNeuralNetworkModel
