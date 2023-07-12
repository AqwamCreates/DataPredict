local GradientDescentModes = {}

local function breakFeatureMatrixToBatches(featureMatrix, labelVector, batchSize, isLabelRequired)
	
	local numberOfBatches = math.ceil(#featureMatrix/batchSize)
	
	local featureMatrixBatchesTable = {}
	
	local labelVectorBatchesTable = {}
	
	local batchPositions = {}
	
	local batchFeatureMatrix
	
	local batchLabelVector 
	
	for batch = 1, numberOfBatches, 1 do
		
		local startIndex = (batch - 1) * batchSize + 1
		
		local endIndex = math.min(batch * batchSize, #featureMatrix)
		
		local batchFeatureMatrix = {}
		
		for i = startIndex, endIndex do table.insert(batchFeatureMatrix, featureMatrix[i]) end
		
		table.insert(featureMatrixBatchesTable, batchFeatureMatrix)
		
		if (isLabelRequired == false) then continue end
		
		batchLabelVector  = {}
		
		for j = startIndex, endIndex do table.insert(batchLabelVector, labelVector[j])	end
		
		table.insert(labelVectorBatchesTable, batchLabelVector)
		
	end
	
	return featureMatrixBatchesTable, labelVectorBatchesTable
	
end

local function startBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector)
	
	MachineLearningModel:train(featureMatrix, labelVector)
	
end

local function startMiniBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector, batchSize, isLabelRequired, showOutputCost)
	
	if (batchSize < 0) then error("Batch size cannot be negative!") end
	
	if (batchSize > #featureMatrix) then error("Batch size is greater than the number of data!") end
	
	local numberOfBatches = math.ceil(#featureMatrix/batchSize)
	
	local featureMatrixBatchesTable, labelVectorBatchesTable = breakFeatureMatrixToBatches(featureMatrix, labelVector, batchSize, isLabelRequired)
	
	local batchFeatureMatrix
	
	local batchLabelVector
	
	local dataArray
	
	local costArray
	
	local cost
	
	for currentBatchNumber = 1, numberOfBatches, 1 do

		batchFeatureMatrix = featureMatrixBatchesTable[currentBatchNumber]

		batchLabelVector = labelVectorBatchesTable[currentBatchNumber]

		dataArray = {batchFeatureMatrix, batchLabelVector}

		costArray = MachineLearningModel:train(featureMatrix, labelVector)
		
		cost = costArray[#costArray]
		
		if (showOutputCost) then print("Epoch: " .. currentBatchNumber .. "\t\t\tFinal Cost: " .. cost .. "\n") end
		
	end

end

local function startStochasticGradientDescent(MachineLearningModel, featureMatrix, labelVector, isLabelRequired, showOutputCost)
	
	local featureVector
	
	local label
	
	local costArray
	
	local cost
	
	for dataIndex = 1, #featureMatrix, 1 do
		
		featureVector = {featureMatrix[dataIndex]}
		
		if (isLabelRequired) then label = {labelVector[dataIndex]} end
		
		costArray = MachineLearningModel:train(featureVector, label)
		
		cost = costArray[#costArray]
		
		if (showOutputCost) then print("Data Number: " .. dataIndex .. "\t\tFinal Cost: " .. cost .. "\n") end
		
	end

end

function GradientDescentModes:startGradientDescent(MachineLearningModel, gradientDescentAlgorithmType, featureMatrix, labelVector, batchSize, isLabelRequired, showOutputCost)
	
	if (typeof(isLabelRequired) == "nil") then isLabelRequired = true end
	
	if (typeof(showOutputCost) == "nil") then showOutputCost = true end
	
	if (gradientDescentAlgorithmType == "Batch") then
		
		startBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector)
		
	elseif (gradientDescentAlgorithmType == "MiniBatch") then
		
		batchSize = batchSize or 2
		
		startMiniBatchGradientDescent(MachineLearningModel, featureMatrix, labelVector, batchSize, isLabelRequired, showOutputCost)
		
	elseif (gradientDescentAlgorithmType == "Stochastic") then
		
		startStochasticGradientDescent(MachineLearningModel, featureMatrix, labelVector, isLabelRequired, showOutputCost)
		
	else
		
		error("The Selected Gradient Descent Algorithm Type Cannot Be Found!")
		
	end
	
end

return GradientDescentModes
