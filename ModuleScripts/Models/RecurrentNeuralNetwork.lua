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

function RecurrentNeuralNetworkModel.new(tokenSize, hiddenSize, maxNumberOfIterations, learningRate, activationFunction, targetCost)

	local NewRecurrentNeuralNetworkModel = BaseModel.new()

	setmetatable(NewRecurrentNeuralNetworkModel, RecurrentNeuralNetworkModel)

	NewRecurrentNeuralNetworkModel.tokenSize = tokenSize
	
	NewRecurrentNeuralNetworkModel.hiddenSize = hiddenSize

	NewRecurrentNeuralNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewRecurrentNeuralNetworkModel.learningRate = learningRate or defaultLearningRate

	NewRecurrentNeuralNetworkModel.activationFunction = activationFunction or defaultActivationFunction

	NewRecurrentNeuralNetworkModel.targetCost = targetCost or defaultTargetCost

	return NewRecurrentNeuralNetworkModel

end

function RecurrentNeuralNetworkModel:setParameters(tokenSize, hiddenSize, maxNumberOfIterations, learningRate, activationFunction, targetCost)

	self.tokenSize = tokenSize or self.tokenSize
	
	self.hiddenSize = hiddenSize or self.hiddenSize

	if (tokenSize) or (hiddenSize) then

		self.ModelParameters = nil

	end

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.activationFunction = activationFunction or self.activationFunction

	self.targetCost = targetCost or self.targetCost

end

function RecurrentNeuralNetworkModel:convertTokenToLogisticVector(token)

	local numberOfNeurons = self.tokenSize

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(numberOfNeurons, 1)

	if (token ~= nil) then

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

function RecurrentNeuralNetworkModel:train(tokenInputSequenceArray, tokenOutputSequenceArray)

	if (self.ModelParameters) then

		self:loadModelParameters()

	else

		self.Wax = self:initializeMatrixBasedOnMode(self.hiddenSize, self.tokenSize)

		self.Waa = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize)

		self.Wya = self:initializeMatrixBasedOnMode(self.tokenSize, self.hiddenSize)

		self.ba = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		self.by = AqwamMatrixLibrary:createMatrix(self.tokenSize, 1)

	end

	local previousdWax

	local previousdWaa

	local previousdWya

	local previousdby

	local previousdba

	local tokenInputSequenceLength = #tokenInputSequenceArray

	local tokenOutputSequenceLength

	if (tokenOutputSequenceArray) then

		tokenOutputSequenceLength = #tokenOutputSequenceArray

	else

		tokenOutputSequenceLength = 0

	end

	local numberOfIterations = 0

	local costArray = {}

	local xTable = {}

	local yTable = {}

	for t = 1, tokenInputSequenceLength, 1 do

		local tokenInput = tokenInputSequenceArray[t]

		local xt = self:convertTokenToLogisticVector(tokenInput)

		table.insert(xTable, xt)

	end

	if (tokenOutputSequenceArray) then

		for t = 1, tokenOutputSequenceLength, 1 do

			local tokenOutput = tokenOutputSequenceArray[t]

			local yt = self:convertTokenToLogisticVector(tokenOutput)

			table.insert(yTable, yt)

		end


	end

	repeat

		numberOfIterations += 1

		local cost = 0

		local dWax = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.tokenSize)

		local dWaa = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize)

		local dWya = AqwamMatrixLibrary:createMatrix(self.tokenSize, self.hiddenSize)

		local dba = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local dby = AqwamMatrixLibrary:createMatrix(self.tokenSize, 1)

		local dx = {}

		local aTable = {}

		local ytPredictionTable = {}

		local daTable = {}

		local tokenInput

		local xt

		local aPrevious = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)

		local aPreviousFirst = aPrevious

		local aNext

		local ytPrediction

		local daNext

		local dxt

		local daPreviousT

		local dWaxt

		local dWaat

		local dWyat

		local dbat

		local dat

		for t = 1, tokenInputSequenceLength, 1 do

			xt = xTable[t]

			aNext = self:forwardPropagateCell(xt, aPrevious)

			ytPrediction = self:calculatePrediction(aNext)

			dat = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

			aPrevious = aNext

			table.insert(aTable, aNext)

			table.insert(ytPredictionTable, ytPrediction)

			table.insert(daTable, dat)

		end

		for t = tokenInputSequenceLength, 1, -1 do

			if (t > 1) then

				aPrevious = aTable[t-1]

			else

				aPrevious = aPreviousFirst

			end

			ytPrediction = ytPredictionTable[t]

			xt = xTable[t]

			aNext = aTable[t]

			daNext = daTable[t]

			dxt, daPreviousT, dWaxt, dWaat, dbat = self:backwardPropagateCell(daNext, aNext, aPrevious, xt) -- daNext for some reason is always is zero Matrix

			daTable[t-1] = AqwamMatrixLibrary:add(daNext, daPreviousT)

			dWax = AqwamMatrixLibrary:add(dWax, dWaxt)

			dWaa = AqwamMatrixLibrary:add(dWaa, dWaat)

			dba = AqwamMatrixLibrary:add(dba, dbat)

			if (tokenOutputSequenceLength > 0) then

				local yt = yTable[t]

				dWyat = AqwamMatrixLibrary:subtract(ytPrediction, yt)

			else

				dWyat = AqwamMatrixLibrary:subtract(ytPrediction, xt)

			end

			dWya = AqwamMatrixLibrary:add(dWya, dWyat)

			cost = cost + AqwamMatrixLibrary:sum(dWya)

		end
		
		dWax = AqwamMatrixLibrary:multiply(self.learningRate, dWax)

		dWaa = AqwamMatrixLibrary:multiply(self.learningRate, dWaa)

		dWya = AqwamMatrixLibrary:multiply(self.learningRate, dWya)

		dba = AqwamMatrixLibrary:multiply(self.learningRate, dba)

		dxt = AqwamMatrixLibrary:multiply(self.learningRate, dxt)

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

		self.Wax = AqwamMatrixLibrary:add(self.Wax, dWax)

		self.Waa = AqwamMatrixLibrary:add(self.Waa, dWaa)

		self.Wya = AqwamMatrixLibrary:add(self.Wya, dWya)

		self.ba = AqwamMatrixLibrary:add(self.ba, dba)

		self.by = AqwamMatrixLibrary:add(self.by, dxt)

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

	local aPrevious = AqwamMatrixLibrary:createRandomNormalMatrix(self.tokenSize, 1)

	local predictionArray = {}

	for i = 1, #tokenInputSequenceArray, 1 do

		local tokenInput = tokenInputSequenceArray[i]

		local xt = self:convertTokenToLogisticVector(tokenInput)

		local aNext = self:forwardPropagateCell(xt, aPrevious)

		local ytPrediction = self:calculatePrediction(aNext)

		local _, predictedTokenIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(ytPrediction)

		local predictedToken = predictedTokenIndex[1]

		table.insert(predictionArray, predictedToken)

		aPrevious = aNext

	end

	return predictionArray

end

return RecurrentNeuralNetworkModel
