local BaseModel = require(script.Parent.BaseModel)

LongShortTermMemoryModel = {}

LongShortTermMemoryModel.__index = LongShortTermMemoryModel

setmetatable(LongShortTermMemoryModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.0001

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

function LongShortTermMemoryModel.new(maxNumberOfIterations, learningRate, targetCost)
	
	local NewLongShortTermMemoryModel = BaseModel.new()

	setmetatable(NewLongShortTermMemoryModel, LongShortTermMemoryModel)
	
	NewLongShortTermMemoryModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewLongShortTermMemoryModel.learningRate = learningRate or defaultLearningRate

	NewLongShortTermMemoryModel.targetCost = targetCost or defaultTargetCost
	
	return NewLongShortTermMemoryModel
	
end

function LongShortTermMemoryModel:setParameters(maxNumberOfIterations, learningRate, targetCost)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost
	
end

function LongShortTermMemoryModel:convertTokenToLogisticVector(token)

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)
	
	if (token ~= nil) then
		
		logisticMatrix[token][1] = 1
		
	end
	
	return logisticMatrix

end

function LongShortTermMemoryModel:forwardPropagateCell(xt, aPrevious, cPrevious)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local concat = AqwamMatrixLibrary:verticalConcatenate(aPrevious, xt)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local ftPart1 = AqwamMatrixLibrary:dotProduct(self.Wf, concat)
	
	local ftPart2 = AqwamMatrixLibrary:add(ftPart1, self.bf)
	
	local ft = AqwamMatrixLibrary:applyFunction(activationFunctionList["sigmoid"], ftPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local itPart1 = AqwamMatrixLibrary:dotProduct(self.Wi, concat)
	
	local itPart2 = AqwamMatrixLibrary:add(itPart1, self.bi)
	
	local it = AqwamMatrixLibrary:applyFunction(activationFunctionList["sigmoid"], itPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local cctPart1 = AqwamMatrixLibrary:dotProduct(self.Wc, concat)
	
	local cctPart2 = AqwamMatrixLibrary:add(cctPart1, self.bc)
	
	local cct = AqwamMatrixLibrary:applyFunction(activationFunctionList["tanh"], cctPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local cNextPart1 = AqwamMatrixLibrary:multiply(ft, cPrevious)
	
	local cNextPart2 = AqwamMatrixLibrary:multiply(it, cct)
	
	local cNext = AqwamMatrixLibrary:add(cNextPart1, cNextPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local otPart1 = AqwamMatrixLibrary:dotProduct(self.Wo, concat)
	
	local otPart2 = AqwamMatrixLibrary:add(otPart1, self.bo)
	
	local ot = AqwamMatrixLibrary:applyFunction(activationFunctionList["sigmoid"], otPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local aNextPart1 = AqwamMatrixLibrary:applyFunction(activationFunctionList["tanh"], cNext)
	
	local aNext = AqwamMatrixLibrary:multiply(ot, aNextPart1)
	
	------------------------------------------------------------------------------------------------------------------------
	
	return aNext, cNext, ft, it, cct, ot
	
end

function LongShortTermMemoryModel:calculatePrediction(aNext)
	
	local ytPredictionPart1 = AqwamMatrixLibrary:dotProduct(self.Wy, aNext)
	
	local ytPredictionPart2 = AqwamMatrixLibrary:add(ytPredictionPart1, self.by)

	local ytPrediction = softMax(ytPredictionPart2)
	
	return ytPrediction
	
end

function LongShortTermMemoryModel:backwardPropagateCell(daNext, dcNext, aNext, cNext, aPrevious, cPrevious, ft, it, cct, ot, xt)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local dtanh = AqwamMatrixLibrary:applyFunction(activationFunctionList["tanh"], cNext)
	
	------------------------------------------------------------------------------------------------------------------------

	local dtanh2 = AqwamMatrixLibrary:applyFunction(derivativeList["tanh"], cNext)
	
	------------------------------------------------------------------------------------------------------------------------

	local dotPart1 = AqwamMatrixLibrary:subtract(1, ot)
	
	local dot = AqwamMatrixLibrary:multiply(daNext, dtanh, ot, dotPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local dcctPart1 = AqwamMatrixLibrary:power(cct, 2)
	
	local dcctPart2 = AqwamMatrixLibrary:subtract(1, dcctPart1)
	
	local dcctPart3 = AqwamMatrixLibrary:multiply(dcNext, it)
	
	local dcctPart4 = AqwamMatrixLibrary:multiply(ot, dtanh2, it, daNext)
	
	local dcctPart5 = AqwamMatrixLibrary:add(dcctPart3, dcctPart4)
	
	local dcct = AqwamMatrixLibrary:multiply(dcctPart5, dcctPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local ditPart1 = AqwamMatrixLibrary:subtract(1, it)
	
	local ditPart2 = AqwamMatrixLibrary:multiply(ot, dtanh2, cct, daNext)
	
	local ditPart3 = AqwamMatrixLibrary:multiply(dcNext, cct)
	
	local ditPart4 = AqwamMatrixLibrary:add(ditPart3, ditPart2)
	
	local dit = AqwamMatrixLibrary:multiply(ditPart4, it, ditPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local dftPart1 = AqwamMatrixLibrary:subtract(1, ft)
	
	local dftPart2 = AqwamMatrixLibrary:multiply(ot, dtanh2, cPrevious, daNext)
	
	local dftPart3 = AqwamMatrixLibrary:multiply(dcNext, cPrevious)
	
	local dftPart4 = AqwamMatrixLibrary:add(dftPart3, dftPart2)
	
	local dft = AqwamMatrixLibrary:multiply(dftPart4, ft, dftPart1) -- (h, 1)
	
	------------------------------------------------------------------------------------------------------------------------

	local concatPart1 = AqwamMatrixLibrary:verticalConcatenate(aPrevious, xt)
	
	local concat = AqwamMatrixLibrary:transpose(concatPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local dWf = AqwamMatrixLibrary:dotProduct(dft, concat) -- (h, h + i)
	
	local dWi = AqwamMatrixLibrary:dotProduct(dit, concat)
	
	local dWc = AqwamMatrixLibrary:dotProduct(dcct, concat)
	
	local dWo = AqwamMatrixLibrary:dotProduct(dot, concat)
	
	local dbf = AqwamMatrixLibrary:sum(dft)
	
	local dbi = AqwamMatrixLibrary:sum(dit)
	
	local dbc = AqwamMatrixLibrary:sum(dcct)
	
	local dbo = AqwamMatrixLibrary:sum(dot)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local WfTransposed = AqwamMatrixLibrary:transpose(self.Wf) -- (h + i, h)
	
	local WiTransposed = AqwamMatrixLibrary:transpose(self.Wi)
	
	local WcTransposed = AqwamMatrixLibrary:transpose(self.Wc)
	
	local WoTransposed = AqwamMatrixLibrary:transpose(self.Wo)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local WfTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WfTransposed, 0, self.hiddenSize)
	
	local WiTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WiTransposed, 0, self.hiddenSize)
	
	local WcTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WcTransposed, 0, self.hiddenSize)
	
	local WoTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WoTransposed, 0, self.hiddenSize)
	
	local daPreviousPart1 = AqwamMatrixLibrary:dotProduct(WfTransposedExtracted1, dft) -- (h + i, 1)
	
	local daPreviousPart2 = AqwamMatrixLibrary:dotProduct(WiTransposedExtracted1, dit)
	
	local daPreviousPart3 = AqwamMatrixLibrary:dotProduct(WcTransposedExtracted1, dcct)
	
	local daPreviousPart4 = AqwamMatrixLibrary:dotProduct(WoTransposedExtracted1, dot)
	
	local daPrevious = AqwamMatrixLibrary:add(daPreviousPart1, daPreviousPart2, daPreviousPart3, daPreviousPart4)
	
	------------------------------------------------------------------------------------------------------------------------

	local dcPreviousPart1 = AqwamMatrixLibrary:multiply(dcNext, ft)
	
	local dcPreviousPart2 = AqwamMatrixLibrary:multiply(ot, dtanh2, ft, daNext)
	
	local dcPrevious = AqwamMatrixLibrary:add(dcPreviousPart1, dcPreviousPart2)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local WfTransposedExtracted2 = AqwamMatrixLibrary:extractRows(WfTransposed, self.hiddenSize, nil)

	local WiTransposedExtracted2 = AqwamMatrixLibrary:extractRows(WiTransposed, self.hiddenSize, nil)

	local WcTransposedExtracted2 = AqwamMatrixLibrary:extractRows(WcTransposed, self.hiddenSize, nil)

	local WoTransposedExtracted2 = AqwamMatrixLibrary:extractRows(WoTransposed, self.hiddenSize, nil)
	
	local dxtPart1 = AqwamMatrixLibrary:dotProduct(WfTransposedExtracted2, dft)
	
	local dxtPart2 = AqwamMatrixLibrary:dotProduct(WiTransposedExtracted2, dit)
	
	local dxtPart3 = AqwamMatrixLibrary:dotProduct(WcTransposedExtracted2, dcct)
	
	local dxtPart4 = AqwamMatrixLibrary:dotProduct(WoTransposedExtracted2, dot)
	
	local dxt = AqwamMatrixLibrary:add(dxtPart1, dxtPart2, dxtPart3, dxtPart4)
	
	------------------------------------------------------------------------------------------------------------------------
	
	return dxt, daPrevious, dcPrevious, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo
	
end

function LongShortTermMemoryModel:createLayers(inputSize, hiddenSize, outputSize)
	
	self.inputSize = inputSize or self.inputSize

	self.hiddenSize = hiddenSize or self.hiddenSize

	self.outputSize = outputSize or self.outputSize

	if (inputSize == nil) and (hiddenSize == nil) and (outputSize == nil) then return nil end
	
	self.ModelParameters = nil
	
end

function LongShortTermMemoryModel:loadModelParameters()
	
	self.Wf = self.ModelParameters[1]

	self.bf = self.ModelParameters[2]

	self.Wi = self.ModelParameters[3]

	self.bi = self.ModelParameters[4]

	self.Wc = self.ModelParameters[5]
	
	self.bc = self.ModelParameters[6]
	
	self.Wo = self.ModelParameters[7]
	
	self.bo = self.ModelParameters[8]
	
	self.Wy = self.ModelParameters[9]
	
	self.by = self.ModelParameters[10]
	
end

function LongShortTermMemoryModel:train(tokenInputSequenceArray, tokenOutputSequenceArray)

	if (self.ModelParameters) then
		
		self:loadModelParameters()

	else
		
		self.Wf = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bf = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		self.Wi = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bi = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		self.Wc = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bc = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		self.Wo = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bo = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		self.Wy = self:initializeMatrixBasedOnMode(self.outputSize, self.hiddenSize)

		self.by = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)

	end
	
	local tokenInputSequenceLength = #tokenInputSequenceArray
	
	local tokenOutputSequenceLength
	
	if (tokenOutputSequenceArray) then
		
		tokenOutputSequenceLength = #tokenOutputSequenceArray
		
		if (tokenInputSequenceLength ~= tokenOutputSequenceLength) then error("The length of token input and output sequence arrays are not equal!") end
		
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
		
		local partialCost = 0
		
		local dWf = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbf = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1, 1)

		local dWi = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbi = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1, 1)

		local dWc = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbc = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1, 1)

		local dWo = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbo = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1, 1)

		local dWy = AqwamMatrixLibrary:createMatrix(self.outputSize, self.hiddenSize + self.inputSize)

		local dby = AqwamMatrixLibrary:createMatrix(self.outputSize, 1, 1)
		
		local dx = {}
		
		local aTable = {}
		
		local cTable = {}

		local ytPredictionTable = {}

		local daTable = {}
		
		local dcTable = {}
		
		local fTable = {}
		
		local iTable = {}
		
		local ccTable = {}
		
		local oTable = {}

		local tokenInput

		local xt

		local aFirst = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)
		
		local cFirst = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)

		local aPrevious = aFirst
		
		local cPrevious = cFirst
		
		local cNext
		
		local ft
		
		local it
		
		local cct
		
		local ot

		local aNext

		local ytPrediction

		local daNext
		
		local dcNext

		local dxt

		local daPrevious
		
		local dcPrevious

		local dWft
		
		local dbft
		
		local dWit
		
		local dbit
		
		local dWct
		
		local dbct
		
		local dWot
		
		local dbot

		local dat
		
		local dct
		
		local dWyt
		
		local Wyt
		
		for t = 1, tokenInputSequenceLength, 1 do

			xt = xTable[t]

			aNext, cNext, ft, it, cct, ot = self:forwardPropagateCell(xt, aPrevious, cPrevious)

			ytPrediction = self:calculatePrediction(aNext)

			dat = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)
			
			dct = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

			aPrevious = aNext

			table.insert(aTable, aNext)

			table.insert(ytPredictionTable, ytPrediction)
			
			table.insert(cTable, cNext)
			
			table.insert(fTable, ft)
			
			table.insert(iTable, it)
			
			table.insert(ccTable, cct)
			
			table.insert(oTable, ot)

			table.insert(daTable, dat)
			
			table.insert(dcTable, dct)

		end

		for t = tokenInputSequenceLength, 1, -1 do

			if (t > 1) then

				aPrevious = aTable[t-1]
				
				cPrevious = cTable[t-1]

			else

				aPrevious = aFirst
				
				cPrevious = cFirst

			end

			ytPrediction = ytPredictionTable[t]

			xt = xTable[t]

			aNext = aTable[t]
			
			daNext = daTable[t] -- 15 x 1
			
			dcNext = dcTable[t] 
			
			cNext = cTable[t]
			
			ft = fTable[t]
			
			it = iTable[t]
			
			cct = ccTable[t]
			
			ot = oTable[t]

			dxt, daPrevious, dcPrevious, dWft, dbft, dWit, dbit, dWct, dbct, dWot, dbot = self:backwardPropagateCell(daNext, dcNext, aNext, cNext, aPrevious, cPrevious, ft, it, cct, ot, xt)
			
			if (t > 1) then 
				
				daTable[t-1] = AqwamMatrixLibrary:add(daNext, daPrevious) 
				
				dcTable[t-1] = AqwamMatrixLibrary:add(dcNext, dcPrevious)
				
			end

			dWf = AqwamMatrixLibrary:add(dWf, dWft)

			dbf = AqwamMatrixLibrary:add(dbf, dbft)

			dWi = AqwamMatrixLibrary:add(dWi, dWit)

			dbi = AqwamMatrixLibrary:add(dbi, dbit)

			dWc = AqwamMatrixLibrary:add(dWc, dWct)

			dbc = AqwamMatrixLibrary:add(dbc, dbct)

			dWo = AqwamMatrixLibrary:add(dWo, dWot)

			dbo = AqwamMatrixLibrary:add(dbo, dbot)
			
			if (tokenOutputSequenceLength > 0) then
				
				local yt = yTable[t]
				
				dWyt = AqwamMatrixLibrary:subtract(ytPrediction, yt)
				
			else
				
				dWyt = AqwamMatrixLibrary:subtract(ytPrediction, xt)
				
			end

			dWy = AqwamMatrixLibrary:add(dWy, dWyt)

			partialCost = AqwamMatrixLibrary:sum(dWy) / self.outputSize

			cost = cost + partialCost

		end
		
		cost = cost / tokenInputSequenceLength
		
		dWf = AqwamMatrixLibrary:multiply(self.learningRate, dWf)
		
		dbf = AqwamMatrixLibrary:multiply(self.learningRate, dbf)
		
		dWi = AqwamMatrixLibrary:multiply(self.learningRate, dWi)
		
		dbi = AqwamMatrixLibrary:multiply(self.learningRate, dbi)
		
		dWc = AqwamMatrixLibrary:multiply(self.learningRate, dWc)
		
		dbc = AqwamMatrixLibrary:multiply(self.learningRate, dbc)
		
		dWo = AqwamMatrixLibrary:multiply(self.learningRate, dWo)
		
		dbo = AqwamMatrixLibrary:multiply(self.learningRate, dbo)
		
		dWy = AqwamMatrixLibrary:multiply(self.learningRate, dWy)
		
		dWy = AqwamMatrixLibrary:extractColumns(dWy, 0, self.hiddenSize)
		
		dby = AqwamMatrixLibrary:multiply(self.learningRate, dby)

		self.Wf = AqwamMatrixLibrary:add(self.Wf, dWf)
		
		self.bf = AqwamMatrixLibrary:add(self.bf, dbf)
		
		self.Wi = AqwamMatrixLibrary:add(self.Wi, dWi)
		
		self.bi = AqwamMatrixLibrary:add(self.bi, dbi)
		
		self.Wc = AqwamMatrixLibrary:add(self.Wc, dWc)
		
		self.bc = AqwamMatrixLibrary:add(self.bc, dbc)
		
		self.Wo = AqwamMatrixLibrary:add(self.Wo, dWo)
		
		self.bo = AqwamMatrixLibrary:add(self.bo, dbo)
		
		self.Wy = AqwamMatrixLibrary:add(self.Wy, dWy)
		
		self.by = AqwamMatrixLibrary:add(self.by, dby)

		self.ModelParameters = {self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo, self.Wy, self.by}
		
		cost = math.abs(cost)
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
	until (numberOfIterations == self.maxNumberOfIterations) or (cost <= self.targetCost)
	
	return costArray
	
end

function LongShortTermMemoryModel:predict(tokenInputSequenceArray)
	
	if (self.ModelParameters == nil) then error("No Model Parameters Found!") end
	
	self:loadModelParameters()
	
	local cPrevious = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)
	
	local aPrevious = AqwamMatrixLibrary:createRandomNormalMatrix(self.hiddenSize, 1)
	
	local predictionArray = {}
	
	for i = 1, #tokenInputSequenceArray, 1 do
		
		local tokenInput = tokenInputSequenceArray[i]
		
		local xt = self:convertTokenToLogisticVector(tokenInput)
		
		local aNext, cNext = self:forwardPropagateCell(xt, aPrevious, cPrevious)
		
		local ytPrediction = self:calculatePrediction(aNext)
		
		local _, predictedTokenIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(ytPrediction)
		
		local predictedToken = predictedTokenIndex[1]
			
		table.insert(predictionArray, predictedToken)
		
		aPrevious = aNext
		
		cPrevious = cNext
		
	end
	
	return predictionArray
	
end

return LongShortTermMemoryModel
