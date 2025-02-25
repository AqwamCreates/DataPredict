--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local BaseInstance = require("Core_BaseInstance")

local TrainingModifier = {}

TrainingModifier.__index = TrainingModifier

setmetatable(TrainingModifier, BaseIntstance)

local defaultTrainingMode = "Stochastic"

local defaultBatchSize = 2

local defaultIsOutputPrinted = true

function TrainingModifier.new(parameterDictionary)
	
	local NewTrainingModifier = BaseIntstance.new(parameterDictionary)
	
	setmetatable(NewTrainingModifier, TrainingModifier)
	
	NewTrainingModifier:setName("TrainingModifier")
	
	NewTrainingModifier:setClassName("TrainingModifier")
	
	NewTrainingModifier.trainingMode = parameterDictionary.trainingMode or defaultTrainingMode

	NewTrainingModifier.batchSize = parameterDictionary.batchSize or defaultBatchSize
	
	NewTrainingModifier.isOutputPrinted = NewTrainingModifier:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, defaultIsOutputPrinted)
	
	NewTrainingModifier.Model = parameterDictionary.Model
	
	return NewTrainingModifier
	
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

function TrainingModifier:batchTrain(...)
	
	return self.Model:train(...)
	
end

function TrainingModifier:miniBatchTrain(...)
	
	local batchSize = self.batchSize
	
	if (batchSize < 0) then error("Batch size cannot be negative!") end
	
	local matrixArray = {...}
	
	local numberOfMatrices = #matrixArray

	local numberOfData = #matrixArray[1]

	for matrixIndex = 1, numberOfMatrices, 1 do

		if (numberOfData ~= #matrixArray[matrixIndex]) then error("All matrices or vectors must contain same number of data") end

	end
	
	if (batchSize > numberOfData) then error("Batch size is greater than the number of data!") end
	
	local numberOfBatches = math.ceil(numberOfData / batchSize)
	
	local miniBatchMatrixArray = {}
	
	for matrixIndex = 1, numberOfMatrices, 1 do
		
		local matrices = breakMatrixToMultipleSmallerMatrices(matrixArray[matrixIndex], batchSize)
		
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
		
		if (self.isOutputPrinted) then print("Epoch: " .. currentBatchNumber .. "\t\t\tFinal cost: " .. cost) end
		
	end
	
	return costArray

end

function TrainingModifier:stochasticTrain(...)
	
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
		
		if (self.isOutputPrinted) then print("Data number: " .. dataIndex .. "\t\tFinal cost: " .. cost) end
		
	end
	
	return costArray

end

function TrainingModifier:train(...)
	
	local trainingMode = self.trainingMode
	
	if (trainingMode == "Batch") then
		
		return self:batchTrain(...)
		
	elseif (trainingMode == "MiniBatch") then
		
		return self:miniBatchTrain(...)
		
	elseif (trainingMode == "Stochastic") then
		
		return self:stochasticTrain(...)
		
	else
		
		error("The selected gradient descent method cannot be found.")
		
	end
	
end

function TrainingModifier:predict(...)
	
	return self.Model:predict(...)
	
end

function TrainingModifier:reinforce(...)
	
	return self.Model:reinforce(...)
	
end

function TrainingModifier:setModel(Model)
	
	self.Model = Model
	
end

function TrainingModifier:getModel()

	return self.Model

end

return TrainingModifier