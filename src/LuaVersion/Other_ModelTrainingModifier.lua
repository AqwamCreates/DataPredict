--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local BaseIntstance = require("Core_BaseInstance")

local ModelTrainingModifier = {}

ModelTrainingModifier.__index = ModelTrainingModifier

setmetatable(ModelTrainingModifier, BaseIntstance)

local defaultTrainingMode = "Stochastic"

local defaultBatchSize = 2

local defaultIsOutputPrinted = true

function ModelTrainingModifier.new(parameterDictionary)
	
	local NewModelTrainingModifier = BaseIntstance.new(parameterDictionary)
	
	setmetatable(NewModelTrainingModifier, ModelTrainingModifier)
	
	NewModelTrainingModifier:setName("ModelTrainingModifier")
	
	NewModelTrainingModifier:setClassName("ModelTrainingModifier")
	
	NewModelTrainingModifier.trainingMode = parameterDictionary.trainingMode or defaultTrainingMode

	NewModelTrainingModifier.batchSize = parameterDictionary.batchSize or defaultBatchSize
	
	NewModelTrainingModifier.isOutputPrinted = NewModelTrainingModifier:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, defaultIsOutputPrinted)
	
	NewModelTrainingModifier.Model = parameterDictionary.Model
	
	return NewModelTrainingModifier
	
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

function ModelTrainingModifier:batchTrain(...)
	
	return self.Model:train(...)
	
end

function ModelTrainingModifier:miniBatchTrain(...)
	
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
	
	local Model = self.Model
	
	local isOutputPrinted = self.isOutputPrinted
	
	for currentBatchNumber = 1, numberOfBatches, 1 do
		
		local currentMatrixBatchArray = {}
		
		for matrixIndex = 1, numberOfMatrices, 1 do
			
			table.insert(currentMatrixBatchArray, miniBatchMatrixArray[matrixIndex][currentBatchNumber])
			
		end

		local miniBatchCostArray = Model:train(table.unpack(currentMatrixBatchArray))
		
		local cost = miniBatchCostArray[#miniBatchCostArray]
		
		table.insert(costArray, cost)
		
		if (isOutputPrinted) then print("Batch: " .. currentBatchNumber .. "\t\t\tFinal cost: " .. cost) end
		
	end
	
	return costArray

end

function ModelTrainingModifier:stochasticTrain(...)
	
	local matrixArray = {...}

	local numberOfMatrices = #matrixArray

	local numberOfData = #matrixArray[1]

	for matrixIndex = 1, numberOfMatrices, 1 do

		if (numberOfData ~= #matrixArray[matrixIndex]) then error("All matrices or vectors must contain same number of data") end

	end
	
	local costArray = {}
	
	local Model = self.Model
	
	local isOutputPrinted = self.isOutputPrinted
	
	local originalMaximumNumberOfIterations = Model.maximumNumberOfIterations
	
	Model.maximumNumberOfIterations = 1
	
	for dataIndex = 1, numberOfData, 1 do
		
		local currentMatrixBatchArray = {}

		for matrixIndex = 1, numberOfMatrices, 1 do

			table.insert(currentMatrixBatchArray, {matrixArray[matrixIndex][dataIndex]})
			
		end
		
		local stochasticCostArray = Model:train(table.unpack(currentMatrixBatchArray))
		
		local cost = stochasticCostArray[#stochasticCostArray]
		
		table.insert(costArray, cost)
		
		if (isOutputPrinted) then print("Data number: " .. dataIndex .. "\t\tFinal cost: " .. cost) end
		
	end
	
	Model.maximumNumberOfIterations = originalMaximumNumberOfIterations
	
	return costArray

end

function ModelTrainingModifier:train(...)
	
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

function ModelTrainingModifier:update(...)

	return self.Model:update(...)

end

function ModelTrainingModifier:predict(...)
	
	return self.Model:predict(...)
	
end

function ModelTrainingModifier:setModel(Model)
	
	self.Model = Model
	
end

function ModelTrainingModifier:getModel()

	return self.Model

end

function ModelTrainingModifier:getModelParameters(...)

	return self.Model:getModelParameters(...)

end

function ModelTrainingModifier:setModelParameters(...)

	self.Model:setModelParameters(...)

end

return ModelTrainingModifier
