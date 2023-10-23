--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

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

local function breakFeatureMatrixToBatches(featureMatrix, labelVector, batchSize)
	
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
		
		if (labelVector == nil) then continue end
		
		batchLabelVector  = {}
		
		for j = startIndex, endIndex do table.insert(batchLabelVector, labelVector[j])	end
		
		table.insert(labelVectorBatchesTable, batchLabelVector)
		
	end
	
	return featureMatrixBatchesTable, labelVectorBatchesTable
	
end

function GradientDescentModifier:startBatchGradientDescent(featureMatrix, labelVector)
	
	return self.Model:train(featureMatrix, labelVector)
	
end

function GradientDescentModifier:startMiniBatchGradientDescent(featureMatrix, labelVector)
	
	if (self.batchSize < 0) then error("Batch size cannot be negative!") end
	
	if (self.batchSize > #featureMatrix) then error("Batch size is greater than the number of data!") end
	
	local numberOfBatches = math.ceil(#featureMatrix/self.batchSize)
	
	local featureMatrixBatchesTable, labelVectorBatchesTable = breakFeatureMatrixToBatches(featureMatrix, labelVector, self.batchSize)
	
	local batchFeatureMatrix
	
	local batchLabelVector
	
	local miniBatchCostArray
	
	local cost
	
	local costArray = {}
	
	for currentBatchNumber = 1, numberOfBatches, 1 do

		batchFeatureMatrix = featureMatrixBatchesTable[currentBatchNumber]

		batchLabelVector = labelVectorBatchesTable[currentBatchNumber]

		miniBatchCostArray = self.Model:train(featureMatrix, labelVector)
		
		cost = miniBatchCostArray[#miniBatchCostArray]
		
		table.insert(costArray, costArray)
		
		if (self.showOutput) then print("Epoch: " .. currentBatchNumber .. "\t\t\tFinal Cost: " .. cost .. "\n") end
		
	end
	
	return costArray

end

function GradientDescentModifier:startStochasticGradientDescent(featureMatrix, labelVector)
	
	local featureVector
	
	local label
	
	local stochasticCostArray
	
	local costArray = {}
	
	local cost
	
	for dataIndex = 1, #featureMatrix, 1 do
		
		featureVector = {featureMatrix[dataIndex]}
		
		if (labelVector[dataIndex]) then label = {labelVector[dataIndex]} end
		
		stochasticCostArray = self.Model:train(featureVector, label)
		
		cost = stochasticCostArray[#stochasticCostArray]
		
		table.insert(costArray, cost)
		
		if (self.showOutput) then print("Data Number: " .. dataIndex .. "\t\tFinal Cost: " .. cost .. "\n") end
		
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
		
		error("The Selected Gradient Descent Algorithm Type Cannot Be Found!")
		
	end
	
end

function GradientDescentModifier:predict(featureMatrix, returnOriginalOutput)
	
	return self.Model:predict(featureMatrix, returnOriginalOutput)
	
end

return GradientDescentModifier
