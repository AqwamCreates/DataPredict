local BaseModel = require(script.Parent.BaseModel)

LongShortTermMemoryModel = {}

LongShortTermMemoryModel.__index = LongShortTermMemoryModel

setmetatable(LongShortTermMemoryModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.0001

local defaultTargetCost = 0

local activationFunctionList = {

	["Sigmoid"] = function (z) return 1/(1+math.exp(-1 * z)) end,

	["Tanh"] = function (z) return math.tanh(z) end,

	["ReLU"] = function (z) return math.max(0, z) end,

	["LeakyReLU"] = function (z) return math.max((0.01 * z), z) end,

	["ELU"] = function (z) return if (z > 0) then z else (0.01 * (math.exp(z) - 1)) end

}

local derivativeList = {

	["Sigmoid"] = function (z) 

		local a = activationFunctionList["Sigmoid"](z)

		return (a * (1-a))

	end,

	["Tanh"] = function (z)

		local a = activationFunctionList["Tanh"](z)

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

function LongShortTermMemoryModel.new(maxNumberOfIterations, learningRate)
	
	local NewLongShortTermMemoryModel = BaseModel.new()

	setmetatable(NewLongShortTermMemoryModel, LongShortTermMemoryModel)
	
	NewLongShortTermMemoryModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewLongShortTermMemoryModel.learningRate = learningRate or defaultLearningRate
	
	return NewLongShortTermMemoryModel
	
end

function LongShortTermMemoryModel:setParameters(maxNumberOfIterations, learningRate)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate
	
end

function LongShortTermMemoryModel:convertTokenToLogisticVector(token)

	if (type(token) == nil) then error("A token is not an integer!") end
	
	if (token > self.outputSize) then error("A token value is larger than the number of output neurons!") end

	local logisticMatrix = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)

	if (token ~= 0) then

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
	
	local ft = AqwamMatrixLibrary:applyFunction(activationFunctionList["Sigmoid"], ftPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local itPart1 = AqwamMatrixLibrary:dotProduct(self.Wi, concat)
	
	local itPart2 = AqwamMatrixLibrary:add(itPart1, self.bi)
	
	local it = AqwamMatrixLibrary:applyFunction(activationFunctionList["Sigmoid"], itPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local cctPart1 = AqwamMatrixLibrary:dotProduct(self.Wc, concat)
	
	local cctPart2 = AqwamMatrixLibrary:add(cctPart1, self.bc)
	
	local cct = AqwamMatrixLibrary:applyFunction(activationFunctionList["Tanh"], cctPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local cNextPart1 = AqwamMatrixLibrary:multiply(ft, cPrevious)
	
	local cNextPart2 = AqwamMatrixLibrary:multiply(it, cct)
	
	local cNext = AqwamMatrixLibrary:add(cNextPart1, cNextPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local otPart1 = AqwamMatrixLibrary:dotProduct(self.Wo, concat)
	
	local otPart2 = AqwamMatrixLibrary:add(otPart1, self.bo)
	
	local ot = AqwamMatrixLibrary:applyFunction(activationFunctionList["Sigmoid"], otPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local aNextPart1 = AqwamMatrixLibrary:applyFunction(activationFunctionList["Tanh"], cNext)
	
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
	
	local tanh = AqwamMatrixLibrary:applyFunction(activationFunctionList["Tanh"], cNext)
	
	------------------------------------------------------------------------------------------------------------------------

	local dtanh = AqwamMatrixLibrary:applyFunction(derivativeList["Tanh"], cNext)
	
	------------------------------------------------------------------------------------------------------------------------

	local dotPart1 = AqwamMatrixLibrary:subtract(1, ot)
	
	local dot = AqwamMatrixLibrary:multiply(daNext, tanh, ot, dotPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local dcctPart1 = AqwamMatrixLibrary:power(cct, 2)
	
	local dcctPart2 = AqwamMatrixLibrary:subtract(1, dcctPart1)
	
	local dcctPart3 = AqwamMatrixLibrary:multiply(dcNext, it)
	
	local dcctPart4 = AqwamMatrixLibrary:multiply(ot, dtanh, it, daNext)
	
	local dcctPart5 = AqwamMatrixLibrary:add(dcctPart3, dcctPart4)
	
	local dcct = AqwamMatrixLibrary:multiply(dcctPart5, dcctPart2)
	
	------------------------------------------------------------------------------------------------------------------------

	local ditPart1 = AqwamMatrixLibrary:subtract(1, it)
	
	local ditPart2 = AqwamMatrixLibrary:multiply(ot, dtanh, cct, daNext)
	
	local ditPart3 = AqwamMatrixLibrary:multiply(dcNext, cct)
	
	local ditPart4 = AqwamMatrixLibrary:add(ditPart3, ditPart2)
	
	local dit = AqwamMatrixLibrary:multiply(ditPart4, it, ditPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local dftPart1 = AqwamMatrixLibrary:subtract(1, ft)
	
	local dftPart2 = AqwamMatrixLibrary:multiply(ot, dtanh, cPrevious, daNext)
	
	local dftPart3 = AqwamMatrixLibrary:multiply(dcNext, cPrevious)
	
	local dftPart4 = AqwamMatrixLibrary:add(dftPart3, dftPart2)
	
	local dft = AqwamMatrixLibrary:multiply(dftPart4, ft, dftPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local concatPart1 = AqwamMatrixLibrary:verticalConcatenate(aPrevious, xt)
	
	local concat = AqwamMatrixLibrary:transpose(concatPart1)
	
	------------------------------------------------------------------------------------------------------------------------

	local dWf = AqwamMatrixLibrary:dotProduct(dft, concat)
	
	local dWi = AqwamMatrixLibrary:dotProduct(dit, concat)
	
	local dWc = AqwamMatrixLibrary:dotProduct(dcct, concat)
	
	local dWo = AqwamMatrixLibrary:dotProduct(dot, concat)
	
	local dbf = AqwamMatrixLibrary:sum(dft)
	
	local dbi = AqwamMatrixLibrary:sum(dit)
	
	local dbc = AqwamMatrixLibrary:sum(dcct)
	
	local dbo = AqwamMatrixLibrary:sum(dot)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local WfTransposed = AqwamMatrixLibrary:transpose(self.Wf)
	
	local WiTransposed = AqwamMatrixLibrary:transpose(self.Wi)
	
	local WcTransposed = AqwamMatrixLibrary:transpose(self.Wc)
	
	local WoTransposed = AqwamMatrixLibrary:transpose(self.Wo)
	
	------------------------------------------------------------------------------------------------------------------------
	
	local WfTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WfTransposed, 1, self.hiddenSize)
	
	local WiTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WiTransposed, 1, self.hiddenSize)
	
	local WcTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WcTransposed, 1, self.hiddenSize)
	
	local WoTransposedExtracted1 = AqwamMatrixLibrary:extractRows(WoTransposed, 1, self.hiddenSize)
	
	local daPreviousPart1 = AqwamMatrixLibrary:dotProduct(WfTransposedExtracted1, dft)
	
	local daPreviousPart2 = AqwamMatrixLibrary:dotProduct(WiTransposedExtracted1, dit)
	
	local daPreviousPart3 = AqwamMatrixLibrary:dotProduct(WcTransposedExtracted1, dcct)
	
	local daPreviousPart4 = AqwamMatrixLibrary:dotProduct(WoTransposedExtracted1, dot)
	
	local daPrevious = AqwamMatrixLibrary:add(daPreviousPart1, daPreviousPart2, daPreviousPart3, daPreviousPart4)
	
	------------------------------------------------------------------------------------------------------------------------

	local dcPreviousPart1 = AqwamMatrixLibrary:multiply(dcNext, ft)
	
	local dcPreviousPart2 = AqwamMatrixLibrary:multiply(ot, dtanh, ft, daNext)
	
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

function LongShortTermMemoryModel:setOptimizers(ForgetGateWeightOptimizer, SaveGateWeightOptimizer, TanhWeightOptimizer, FocusGateOptimizer, OutputWeightOptimizer, ForgetGateBiasOptimizer, SaveGateBiasOptimizer, TanhBiasOptimizer, FocusGateBiasOptimizer, OutputBiasOptimizer)

	self.ForgetGateWeightOptimizer = ForgetGateWeightOptimizer

	self.SaveGateWeightOptimizer = SaveGateWeightOptimizer

	self.TanhWeightOptimizer = TanhWeightOptimizer
	
	self.FocusGateOptimizer = FocusGateOptimizer
	
	self.OutputWeightOptimizer = OutputWeightOptimizer
	
	self.ForgetGateBiasOptimizer = ForgetGateBiasOptimizer

	self.SaveGateBiasOptimizer = SaveGateBiasOptimizer

	self.TanhBiasOptimizer = TanhBiasOptimizer

	self.FocusGateBiasOptimizer = FocusGateBiasOptimizer

	self.OutputBiasOptimizer = OutputBiasOptimizer

end

function LongShortTermMemoryModel:setRegularizations(ForgetGateWeightRegularization, SaveGateWeightRegularization, TanhWeightRegularization, FocusGateRegularization, OutputWeightRegularization, ForgetGateBiasRegularization, SaveGateBiasRegularization, TanhBiasRegularization, FocusGateBiasRegularization, OutputBiasRegularization)

	self.ForgetGateWeightRegularization = ForgetGateWeightRegularization

	self.SaveGateWeightRegularization = SaveGateWeightRegularization

	self.TanhWeightRegularization = TanhWeightRegularization

	self.FocusGateRegularization = FocusGateRegularization

	self.OutputWeightRegularization = OutputWeightRegularization

	self.ForgetGateBiasRegularization = ForgetGateBiasRegularization

	self.SaveGateBiasRegularization = SaveGateBiasRegularization

	self.TanhBiasRegularization = TanhBiasRegularization

	self.FocusGateBiasRegularization = FocusGateBiasRegularization

	self.OutputBiasRegularization = OutputBiasRegularization

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

local function throwErrorIfSequenceLengthAreNotEqual(tokenInputSequenceArray, tokenOutputSequenceArray)

	if (tokenOutputSequenceArray == nil) then return nil end

	local tokenInputSequenceLength = #tokenInputSequenceArray

	local tokenOutputSequenceLength = #tokenOutputSequenceArray

	if (tokenInputSequenceLength ~= tokenOutputSequenceLength) then error("The length of token input and output sequence arrays are not equal!") end

end

function LongShortTermMemoryModel:train(tableOfTokenInputSequenceArray, tableOfTokenOutputSequenceArray)

	if (self.ModelParameters) then
		
		self:loadModelParameters()
		
	elseif (self.inputSize == nil) or (self.hiddenSize == nil) or (self.outputSize == nil) then

		error("Layers are not set!")

	else
		
		self.Wf = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bf = self:initializeMatrixBasedOnMode(self.hiddenSize, 1)

		self.Wi = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bi = self:initializeMatrixBasedOnMode(self.hiddenSize, 1)

		self.Wc = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bc = self:initializeMatrixBasedOnMode(self.hiddenSize, 1)

		self.Wo = self:initializeMatrixBasedOnMode(self.hiddenSize, self.hiddenSize + self.inputSize)

		self.bo = self:initializeMatrixBasedOnMode(self.hiddenSize, 1)

		self.Wy = self:initializeMatrixBasedOnMode(self.outputSize, self.hiddenSize)

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
		
		local dWf = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbf = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local dWi = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbi = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local dWc = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbc = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local dWo = AqwamMatrixLibrary:createMatrix(self.hiddenSize, self.hiddenSize + self.inputSize)

		local dbo = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local dWy = AqwamMatrixLibrary:createMatrix(self.outputSize, self.hiddenSize + self.inputSize)

		local dby = AqwamMatrixLibrary:createMatrix(self.outputSize, 1)
		
		for s = 1, #tableOfTokenInputSequenceArray, 1 do
			
			self:dataWait()

			local xTable = tableOfTokenInputSequenceLogisticMatrices[s]

			local yTable = tableOfTokenOutputSequenceLogisticMatrices[s]
			
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
			
			local yt

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

			for t = 1, #xTable, 1 do
				
				self:sequenceWait()

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

			for t = #xTable, 1, -1 do
				
				self:sequenceWait()

				if (t > 1) then

					aPrevious = aTable[t-1]

					cPrevious = cTable[t-1]

				else

					aPrevious = aFirst

					cPrevious = cFirst

				end

				ytPrediction = ytPredictionTable[t]

				xt = xTable[t]
				
				yt = yTable[t]

				aNext = aTable[t]

				daNext = daTable[t]

				dcNext = dcTable[t] 

				cNext = cTable[t]

				ft = fTable[t]

				it = iTable[t]

				cct = ccTable[t]

				ot = oTable[t]

				dxt, daPrevious, dcPrevious, dWft, dbft, dWit, dbit, dWct, dbct, dWot, dbot = self:backwardPropagateCell(daNext, dcNext, aNext, cNext, aPrevious, cPrevious, ft, it, cct, ot, xt)
				
				dWyt = AqwamMatrixLibrary:subtract(ytPrediction, yt)
				
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
				
				dWy = AqwamMatrixLibrary:add(dWy, dWyt)

				partialCost = AqwamMatrixLibrary:sum(dWy)

				cost += partialCost
				
			end
			
		end
		
		cost /= totalNumberOfTokens
		
		dWy = AqwamMatrixLibrary:extractColumns(dWy, 1, self.hiddenSize)

		if (self.ForgetGateWeightRegularization) then

			local regularizationDerivatives = self.ForgetGateWeightRegularization:calculateRegularizationDerivatives(self.Wf, 1)

			dWf = AqwamMatrixLibrary:add(dWf, regularizationDerivatives)

		end

		if (self.SaveGateWeightRegularization) then

			local regularizationDerivatives = self.SaveGateWeightRegularization:calculateRegularizationDerivatives(self.Wi, 1)

			dWi = AqwamMatrixLibrary:add(dWi, regularizationDerivatives)

		end

		if (self.TanhWeightRegularization) then

			local regularizationDerivatives = self.TanhWeightRegularization:calculateRegularizationDerivatives(self.Wc, 1)

			dWc = AqwamMatrixLibrary:add(dWc, regularizationDerivatives)

		end

		if (self.FocusGateRegularization) then

			local regularizationDerivatives = self.FocusGateRegularization:calculateRegularizationDerivatives(self.Wo, 1)

			dWo = AqwamMatrixLibrary:add(dWo, regularizationDerivatives)

		end

		if (self.OutputWeightRegularization) then

			local regularizationDerivatives = self.OutputWeightRegularization:calculateRegularizationDerivatives(self.Wy, 1)

			dWy = AqwamMatrixLibrary:add(dWy, regularizationDerivatives)

		end

		if (self.ForgetGateBiasRegularization) then

			local regularizationDerivatives = self.ForgetGateBiasRegularization:calculateRegularizationDerivatives(self.bf, 1)

			dbf = AqwamMatrixLibrary:add(dbf, regularizationDerivatives)

		end

		if (self.SaveGateBiasRegularization) then

			local regularizationDerivatives = self.SaveGateBiasRegularization:calculateRegularizationDerivatives(self.bi, 1)

			dbi = AqwamMatrixLibrary:add(dbi, regularizationDerivatives)

		end

		if (self.TanhBiasRegularization) then

			local regularizationDerivatives = self.TanhBiasRegularization:calculateRegularizationDerivatives(self.bc, 1)

			dbc = AqwamMatrixLibrary:add(dbc, regularizationDerivatives)

		end

		if (self.FocusGateBiasRegularization) then

			local regularizationDerivatives = self.FocusGateBiasRegularization:calculateRegularizationDerivatives(self.bo, 1)

			dbo = AqwamMatrixLibrary:add(dbo, regularizationDerivatives)

		end

		if (self.OutputBiasRegularization) then

			local regularizationDerivatives = self.OutputBiasRegularization:calculateRegularizationDerivatives(self.by, 1)

			dby = AqwamMatrixLibrary:add(dby, regularizationDerivatives)

		end
		
		--------------------------------------------------------------------------------------------------------
		
		if (self.ForgetGateWeightOptimizer) then
			
			dWf = self.ForgetGateWeightOptimizer:calculate(self.learningRate, dWf)
			
		else
			
			dWf = AqwamMatrixLibrary:multiply(self.learningRate, dWf)
			
		end
		
		if (self.SaveGateWeightOptimizer) then

			dWi = self.SaveGateWeightOptimizer:calculate(self.learningRate, dWi)
			
		else
			
			dWi = AqwamMatrixLibrary:multiply(self.learningRate, dWi)

		end
		
		if (self.TanhWeightOptimizer) then

			dWc = self.TanhWeightOptimizer:calculate(self.learningRate, dWc)
			
		else
			
			dWc = AqwamMatrixLibrary:multiply(self.learningRate, dWc)

		end
		
		if (self.FocusGateOptimizer) then

			dWo = self.FocusGateOptimizer:calculate(self.learningRate, dWo)
			
		else
			
			dWo = AqwamMatrixLibrary:multiply(self.learningRate, dWo)

		end
		
		if (self.OutputWeightOptimizer) then

			dWy = self.OutputWeightOptimizer:calculate(self.learningRate, dWy)
			
		else
			
			dWy = AqwamMatrixLibrary:multiply(self.learningRate, dWy)

		end
		
		if (self.ForgetGateBiasOptimizer) then

			dbf = self.ForgetGateBiasOptimizer:calculate(self.learningRate, dbf)
			
		else
			
			dbf = AqwamMatrixLibrary:multiply(self.learningRate, dbf)

		end
		
		if (self.SaveGateBiasOptimizer) then

			dbi = self.SaveGateBiasOptimizer:calculate(self.learningRate, dbi)
			
		else
			
			dbi = AqwamMatrixLibrary:multiply(self.learningRate, dbi)

		end
		
		if (self.TanhBiasOptimizer) then

			dbc = self.TanhBiasOptimizer:calculate(self.learningRate, dbc)
			
		else
			
			dbc = AqwamMatrixLibrary:multiply(self.learningRate, dbc)

		end
		
		if (self.FocusGateBiasOptimizer) then

			dbo = self.FocusGateBiasOptimizer:calculate(self.learningRate, dbo)
			
		else
			
			dbo = AqwamMatrixLibrary:multiply(self.learningRate, dbo)

		end
		
		if (self.OutputBiasOptimizer) then

			dby = self.OutputBiasOptimizer:calculate(self.learningRate, dby)
			
		else
			
			dby = AqwamMatrixLibrary:multiply(self.learningRate, dby)

		end

		self.Wf = AqwamMatrixLibrary:subtract(self.Wf, dWf)
		
		self.bf = AqwamMatrixLibrary:subtract(self.bf, dbf)
		
		self.Wi = AqwamMatrixLibrary:subtract(self.Wi, dWi)
		
		self.bi = AqwamMatrixLibrary:subtract(self.bi, dbi)
		
		self.Wc = AqwamMatrixLibrary:subtract(self.Wc, dWc)
		
		self.bc = AqwamMatrixLibrary:subtract(self.bc, dbc)
		
		self.Wo = AqwamMatrixLibrary:subtract(self.Wo, dWo)
		
		self.bo = AqwamMatrixLibrary:subtract(self.bo, dbo)
		
		self.Wy = AqwamMatrixLibrary:subtract(self.Wy, dWy)
		
		self.by = AqwamMatrixLibrary:subtract(self.by, dby)

		self.ModelParameters = {self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo, self.Wy, self.by}
		
		cost = math.abs(cost)
		
		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
	until (numberOfIterations == self.maxNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.ForgetGateWeightOptimizer) and (self.AutoResetOptimizers) then

		self.ForgetGateWeightOptimizer:reset()

	end

	if (self.SaveGateWeightOptimizer) and (self.AutoResetOptimizers) then

		self.SaveGateWeightOptimizer:reset()

	end

	if (self.TanhWeightOptimizer) and (self.AutoResetOptimizers) then

		self.TanhWeightOptimizer:reset()

	end

	if (self.FocusGateOptimizer) and (self.AutoResetOptimizers) then

		self.FocusGateOptimizer:reset()

	end

	if (self.OutputWeightOptimizer) and (self.AutoResetOptimizers) then

		self.OutputWeightOptimizer:reset()

	end

	if (self.ForgetGateBiasOptimizer) and (self.AutoResetOptimizers) then

		self.ForgetGateBiasOptimizer:reset()

	end

	if (self.SaveGateBiasOptimizer) and (self.AutoResetOptimizers) then

		self.SaveGateBiasOptimizer:reset()

	end

	if (self.TanhBiasOptimizer) and (self.AutoResetOptimizers) then

		self.TanhBiasOptimizer:reset()

	end

	if (self.FocusGateOptimizer) and (self.AutoResetOptimizers) then

		self.FocusGateOptimizer:reset()

	end

	if (self.OutputBiasOptimizer) and (self.AutoResetOptimizers) then

		self.OutputBiasOptimizer:reset()

	end
	
	return costArray
	
end

function LongShortTermMemoryModel:predict(tableOfTokenInputSequenceLogisticMatrices, returnOriginalOutput)
	
	if (self.ModelParameters == nil) then error("No Model Parameters Found!") end
	
	self:loadModelParameters()
	
	local tableOfTokenOutputSequenceArray = {}
	
	for i = 1, #tableOfTokenInputSequenceLogisticMatrices, 1 do
		
		local tokenInputSequenceArray = tableOfTokenInputSequenceLogisticMatrices[i]
		
		local cPrevious = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local aPrevious = AqwamMatrixLibrary:createMatrix(self.hiddenSize, 1)

		local predictionArray = {}
		
		for j = 1, #tokenInputSequenceArray, 1 do

			local tokenInput = tokenInputSequenceArray[j]

			local xt = self:convertTokenToLogisticVector(self.inputSize, tokenInput)

			local aNext, cNext = self:forwardPropagateCell(xt, aPrevious, cPrevious)

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

			cPrevious = cNext

		end
		
		table.insert(tableOfTokenOutputSequenceArray, predictionArray)
		
	end
	
	return tableOfTokenOutputSequenceArray
	
end

return LongShortTermMemoryModel
