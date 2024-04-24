local GradientDescentModifier = {}

GradientDescentModifier.__index = GradientDescentModifier

local defaultGradientDescentType = "Stochastic"

local defaultBatchSize = 2

local defaultShowOutput = true

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function GradientDescentModifier.new(Model, gradientDescentType, batchSize, showOutput)
	
	local NewGradientDescentModifier = {}
	
	setmetatable(NewGradientDescentModifier, GradientDescentModifier)
	
	NewGradientDescentModifier.Model = Model
	
	NewGradientDescentModifier.gradientDescentType = gradientDescentType or defaultGradientDescentType
	
	NewGradientDescentModifier.batchSize = batchSize or defaultBatchSize
	
	NewGradientDescentModifier.showOutput = getBooleanOrDefaultOption(showOutput, defaultShowOutput)
	
	return NewGradientDescentModifier
	
end

function GradientDescentModifier:setParameters(Model, gradientDescentType, batchSize, showOutput)
	
	self.Model = Model or self.Model

	self.gradientDescentType = gradientDescentType or self.gradientDescentType 

	self.batchSize = batchSize or self.batchSize
	
	self.showOutput = getBooleanOrDefaultOption(showOutput, self.showOutput)
	
end

local function breakMatrixToMultipleSmallerMatrices(matrix, batchSize)
	
	local numberOfBatches = math.ceil(#matrix/batchSize)
	
	local matrixBatchesTable = {}
	
	local batchPositions = {}
	
	local batchFeatureMatrix
	
	local batchLabelVector 
	
	for batch = 1, numberOfBatches, 1 do
		
		local startIndex = (batch - 1) * batchSize + 1
		
		local endIndex = math.min(batch * batchSize, #matrix)
		
		local batchFeatureMatrix = {}
		
		for i = startIndex, endIndex do table.insert(batchFeatureMatrix, matrix[i]) end
		
		table.insert(matrixBatchesTable, batchFeatureMatrix)
		
	end
	
	return matrixBatchesTable
	
end

function GradientDescentModifier:startBatchGradientDescent(...)
	
	return self.Model:train(...)
	
end

function GradientDescentModifier:startMiniBatchGradientDescent(...)
	
	if (self.batchSize < 0) then error("Batch size cannot be negative!") end
	
	local matrixArray = {...}
	
	local numberOfMatrices = #matrixArray

	local numberOfData = #matrixArray[1]

	for matrixIndex = 1, numberOfMatrices, 1 do

		if (numberOfData ~= #matrixArray[matrixIndex]) then error("All matrices or vectors must contain same number of data") end

	end
	
	if (self.batchSize > numberOfData) then error("Batch size is greater than the number of data!") end
	
	local numberOfBatches = math.ceil(numberOfData/self.batchSize)
	
	local miniBatchMatrixArray = {}
	
	for matrixIndex = 1, numberOfMatrices, 1 do
		
		local matrices = breakMatrixToMultipleSmallerMatrices(matrixArray[matrixIndex], self.batchSize)
		
		table.insert(miniBatchMatrixArray, matrices)
		
	end
	
	local costArray = {}
	
	for currentBatchNumber = 1, numberOfBatches, 1 do
		
		local currentMatrixBatchArray = {}
		
		for matrixIndex = 1, numberOfMatrices, 1 do
			
			table.insert(currentMatrixBatchArray, miniBatchMatrixArray[matrixIndex][currentBatchNumber])
			
		end

		local miniBatchCostArray = self.Model:train(table.unpack(currentMatrixBatchArray))
		
		local cost = miniBatchCostArray[#miniBatchCostArray]
		
		table.insert(costArray, costArray)
		
		if (self.showOutput) then print("Epoch: " .. currentBatchNumber .. "\t\t\tFinal cost: " .. cost .. "\n") end
		
	end
	
	return costArray

end

function GradientDescentModifier:startStochasticGradientDescent(...)
	
	local matrixArray = {...}

	local numberOfMatrices = #matrixArray

	local numberOfData = #matrixArray[1]

	for matrixIndex = 1, numberOfMatrices, 1 do

		if (numberOfData ~= #matrixArray[matrixIndex]) then error("All matrices or vectors must contain same number of data") end

	end
	
	local costArray = {}
	
	for dataIndex = 1, numberOfData, 1 do
		
		local currentMatrixBatchArray = {}

		for matrixIndex = 1, numberOfMatrices, 1 do

			table.insert(currentMatrixBatchArray, {matrixArray[matrixIndex][dataIndex]})
			
		end
		
		local stochasticCostArray = self.Model:train(table.unpack(currentMatrixBatchArray))
		
		local cost = stochasticCostArray[#stochasticCostArray]
		
		table.insert(costArray, cost)
		
		if (self.showOutput) then print("Data number: " .. dataIndex .. "\t\tFinal cost: " .. cost .. "\n") end
		
	end
	
	return costArray

end

function GradientDescentModifier:train(featureMatrix, labelVector)
	
	if (self.gradientDescentType == "Batch") then
		
		return self:startBatchGradientDescent(featureMatrix, labelVector)
		
	elseif (self.gradientDescentType == "MiniBatch") then
		
		return self:startMiniBatchGradientDescent(featureMatrix, labelVector)
		
	elseif (self.gradientDescentType == "Stochastic") then
		
		return self:startStochasticGradientDescent(featureMatrix, labelVector)
		
	else
		
		error("The selected gradient descent algorithm type cannot be found.")
		
	end
	
end

function GradientDescentModifier:predict(featureMatrix, returnOriginalOutput)
	
	return self.Model:predict(featureMatrix, returnOriginalOutput)
	
end

function GradientDescentModifier:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
	return self.Model:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
end

return GradientDescentModifier
