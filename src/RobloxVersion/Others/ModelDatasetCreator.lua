ModelDatasetCreator = {}

ModelDatasetCreator.__index = ModelDatasetCreator

local defaultTrainDataPercentage = 0.7

local defaultValidationDataPercentage = 0.2

local defaultTestDataPercentage = 0.1

local defaultIsDatasetRandomized = false

local defaultRandomizationProbabilityThreshold = 0.3

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function returnNilIfTableIsEmpty(tableToCheck)
	
	if (#tableToCheck >= 1) then 
		
		return tableToCheck
		
	else
		
		return nil	
		
	end
	
end

local function checkNumberOfData(featureMatrix, labelVectorOrMatrix)
	
	if (type(labelVectorOrMatrix) ~= "nil") then

		if (#featureMatrix ~= #labelVectorOrMatrix) then error("The feature matrix and the label vector/matrix do not contain the same number of data.") end

	end
	
	return #featureMatrix
	
end

local function deepCopyData(featureMatrix, labelVectorOrMatrix)
	
	featureMatrix = deepCopyTable(featureMatrix)
	
	if (type(labelVectorOrMatrix) ~= "nil") then
		
		labelVectorOrMatrix = deepCopyTable(labelVectorOrMatrix)
		
	end
	
	return featureMatrix, labelVectorOrMatrix
	
end

function ModelDatasetCreator.new()
	
	local NewModelDatasetCreator = {}

	setmetatable(NewModelDatasetCreator, ModelDatasetCreator)
	
	NewModelDatasetCreator.trainDataPercentage = defaultTrainDataPercentage
	
	NewModelDatasetCreator.validationDataPercentage = defaultValidationDataPercentage
	
	NewModelDatasetCreator.testDataPercentage = defaultTestDataPercentage
	
	NewModelDatasetCreator.isDatasetRandomized = defaultIsDatasetRandomized
	
	NewModelDatasetCreator.randomizationProbabilityThreshold = defaultRandomizationProbabilityThreshold
	
	return NewModelDatasetCreator
	
end

function ModelDatasetCreator:setDatasetSplitPercentages(trainDataPercentage, validationDataPercentage, testDataPercentage)
	
	self.trainDataPercentage = trainDataPercentage or self.trainDataPercentage

	self.validationDataPercentage = validationDataPercentage or self.validationDataPercentage

	self.testDataPercentage = testDataPercentage or self.testDataPercentage
	
end

function ModelDatasetCreator:setDatasetRandomizationProperties(isDatasetRandomized, randomizationProbabilityThreshold)
	
	self.isDatasetRandomized = getBooleanOrDefaultOption(isDatasetRandomized, self.isDatasetRandomized)
	
	self.randomizationProbabilityThreshold = randomizationProbabilityThreshold or self.randomizationProbabilityThreshold
	
end

function ModelDatasetCreator:randomizeDataset(featureMatrix, labelVectorOrMatrix)
	
	local numberOfData = checkNumberOfData(featureMatrix, labelVectorOrMatrix)
	
	local randomizationProbabilityThreshold = self.randomizationProbabilityThreshold
	
	local randomizedFeatureMatrix, randomizedLabelVectorOrMatrix = deepCopyTable(featureMatrix, labelVectorOrMatrix)
	
	for index = 1, numberOfData, 1 do
		
		if (math.random() < randomizationProbabilityThreshold) then continue end
		
		local randomIndex = math.random(0, index)
		
		randomIndex = math.ceil(randomIndex)
		
		local temporaryRandomFeatureVector = randomizedFeatureMatrix[index]
		
		table.remove(randomizedFeatureMatrix, index)
		
		table.insert(randomizedFeatureMatrix, temporaryRandomFeatureVector, randomIndex)
		
		if (type(labelVectorOrMatrix) == "nil") then continue end
		
		local temporaryRandomLabelVector = randomizedLabelVectorOrMatrix[index]

		table.remove(randomizedLabelVectorOrMatrix, index)

		table.insert(randomizedLabelVectorOrMatrix, temporaryRandomLabelVector, randomIndex)
		
	end
	
	return randomizedFeatureMatrix, randomizedLabelVectorOrMatrix
	
end

function ModelDatasetCreator:splitDataset(datasetMatrix)
	
	local numberOfData = checkNumberOfData(datasetMatrix)
	
	local datasetCopy = deepCopyTable(datasetMatrix)
	
	local numberOfTrainData = math.floor(self.trainDataPercentage * numberOfData)
	
	local numberOfValidationData = math.floor(self.validationDataPercentage * numberOfData)
	
	local numberOfTestData = math.floor(self.testDataPercentage * numberOfData)
	
	local trainDataMaxValue = numberOfTrainData
	
	local trainValidationDataMaxValue = numberOfTrainData + numberOfValidationData
	
	local trainValidationTestDataMaxValue = trainValidationDataMaxValue + numberOfTestData
	
	local trainData = {}

	local validationData = {}

	local testData = {}
	
	for index = 1, numberOfData, 1 do
		
		if (index < numberOfTrainData) then 
			
			table.insert(trainData, datasetCopy[index])
			continue
			
		end
		
		if (index < trainValidationDataMaxValue) then 

			table.insert(validationData, datasetCopy[index])
			continue

		end
		
		if (index < trainValidationTestDataMaxValue) then 

			table.insert(testData, datasetCopy[index])
			continue

		end
		
		if (numberOfTrainData > 0) then
			
			table.insert(trainData, datasetCopy[index])
			continue
			
		end
		
		if (numberOfValidationData > 0) then

			table.insert(validationData, datasetCopy[index])
			continue

		end
		
		if (numberOfTestData > 0) then

			table.insert(testData, datasetCopy[index])
			continue

		end
		
	end

	trainData = returnNilIfTableIsEmpty(trainData)

	validationData = returnNilIfTableIsEmpty(validationData)

	testData = returnNilIfTableIsEmpty(testData)
	
	table.clear(datasetCopy)
	
	datasetCopy = nil

	return trainData, validationData, testData
	
end

return ModelDatasetCreator
