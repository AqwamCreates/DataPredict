local GradientDescentModes = {}

local function breakFeatureMatrixToBatches(numberOfBatches, featureMatrix, labelVector)
	
	local batchSize = math.floor(#featureMatrix/numberOfBatches)
	
	local remainderBatchSize = #featureMatrix % numberOfBatches
	
	local featureMatrixBatchesTable = {}
	
	local labelVectorBatchesTable = {}
	
	local batchPositions = {}
	
	local newFeatureMatrix
	
	local newLabelVector
	
	for currentBatchNumber = 0, numberOfBatches, 1 do
		
		if (numberOfBatches == currentBatchNumber) then
			
			table.insert(batchPositions, numberOfBatches * (batchSize - 1))
			
		else
			
			table.insert(batchPositions, (currentBatchNumber * batchSize) + 1)
			
		end
		
	end
	
	table.insert(batchPositions, #featureMatrix)
	
	for currentBatchNumber = 1, numberOfBatches, 1 do
		
		newFeatureMatrix = {}
		
		newLabelVector = {}
		
		for row = batchPositions[currentBatchNumber], batchPositions[currentBatchNumber + 1], 1 do
			
			table.insert(newFeatureMatrix, featureMatrix[row])
			
			if (labelVector) then table.insert(newLabelVector, labelVector[row]) end
			
		end
			
		table.insert(featureMatrixBatchesTable, newFeatureMatrix)

		if (labelVector) then table.insert(labelVectorBatchesTable, newLabelVector) end
		
	end
	
	return featureMatrixBatchesTable, labelVectorBatchesTable
	
end

local function startBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector)
	
	MachineLearningModel:train(featureMatrix, labelVector)
	
end

local function startMiniBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector, numberOfBatches)
	
	if (numberOfBatches < 0) then error("Number of batches does not accept negative values!") end
	
	if (numberOfBatches > #featureMatrix) then error("Number of batches is greater than the number of data!") end
	
	local featureMatrixBatchesTable, labelVectorBatchesTable = breakFeatureMatrixToBatches(numberOfBatches, featureMatrix, labelVector)
	
	local batchFeatureMatrix
	
	local batchLabelVector
	
	local dataArray
	
	for currentBatchNumber = 1, numberOfBatches, 1 do

		batchFeatureMatrix = featureMatrixBatchesTable[currentBatchNumber]

		batchLabelVector = labelVectorBatchesTable[currentBatchNumber]

		dataArray = {batchFeatureMatrix, batchLabelVector}

		MachineLearningModel:train(featureMatrix, labelVector)
		
		print("Batch: " .. currentBatchNumber)
		
	end

end

local function startStochasticGradientDescent(MachineLearningModel, featureMatrix, labelVector)
	
	local featureVector
	
	local label
	
	for row = 1, #featureMatrix, 1 do
		
		local featureVector = {featureMatrix[row]}
		
		local label = {labelVector[row]}
		
		MachineLearningModel:train(featureVector, label)
		
	end

end

function GradientDescentModes:startGradientDescent(MachineLearningModel, gradientDescentAlgorithmType, featureMatrix, labelVector, batchSize)
	
	if (gradientDescentAlgorithmType == "Batch") then
		
		startBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector)
		
	elseif (gradientDescentAlgorithmType == "Minibatch") then
		
		batchSize = batchSize or 2
		
		startMiniBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector, batchSize)
		
	elseif (gradientDescentAlgorithmType == "Stochastic") then
		
		startStochasticGradientDescent(MachineLearningModel, featureMatrix, labelVector)
		
	else
		
		error("The Selected Gradient Descent Algorithm Type Cannot Be Found!")
		
	end
	
end

return GradientDescentModes


