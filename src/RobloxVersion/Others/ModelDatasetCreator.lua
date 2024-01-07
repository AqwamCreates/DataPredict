ModelDatasetCreator = {}

ModelDatasetCreator.__index = ModelDatasetCreator

local defaultTrainDataPercentage = 0.7

local defaultValidationDataPercentage = 0

local defaultTestDataPercentage = 0.3

local defaultIsDatasetRandomized = false

local defaultDatasetRandomizationProbability = 0.95

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

function ModelDatasetCreator.new()
	
	local NewModelDatasetCreator = {}

	setmetatable(NewModelDatasetCreator, ModelDatasetCreator)
	
	NewModelDatasetCreator.trainDataPercentage = defaultTrainDataPercentage
	
	NewModelDatasetCreator.validationDataPercentage = defaultValidationDataPercentage
	
	NewModelDatasetCreator.testDataPercentage = defaultTestDataPercentage
	
	NewModelDatasetCreator.datasetRandomizationProbability = defaultDatasetRandomizationProbability
	
	return NewModelDatasetCreator
	
end

function ModelDatasetCreator:setDatasetSplitPercentages(trainDataPercentage, validationDataPercentage, testDataPercentage)
	
	self.trainDataPercentage = trainDataPercentage or self.trainDataPercentage

	self.validationDataPercentage = validationDataPercentage or self.validationDataPercentage

	self.testDataPercentage = testDataPercentage or self.testDataPercentage
	
end

function ModelDatasetCreator:setDatasetRandomizationProbability(datasetRandomizationProbability)
	
	self.datasetRandomizationProbability = datasetRandomizationProbability or self.datasetRandomizationProbability
	
end

function ModelDatasetCreator:randomizeDataset(featureMatrix, labelVectorOrMatrix)
	
	local numberOfData = checkNumberOfData(featureMatrix, labelVectorOrMatrix)
	
	local datasetRandomizationProbability = self.datasetRandomizationProbability
	
	local randomizedFeatureMatrix = deepCopyTable(featureMatrix)
	
	local randomizedLabelVectorOrMatrix = deepCopyTable(labelVectorOrMatrix)
	
	for index = 1, numberOfData, 1 do
		
		if (datasetRandomizationProbability < math.random()) then continue end
		
		local randomIndex = math.random(0, index)
		
		randomIndex = math.ceil(randomIndex)
		
		local temporaryRandomFeatureVector = randomizedFeatureMatrix[index]
		
		table.remove(randomizedFeatureMatrix, index)
		
		table.insert(randomizedFeatureMatrix, randomIndex, temporaryRandomFeatureVector)
		
		if (type(labelVectorOrMatrix) == "nil") then continue end
		
		local temporaryRandomLabelVector = randomizedLabelVectorOrMatrix[index]

		table.remove(randomizedLabelVectorOrMatrix, index)

		table.insert(randomizedLabelVectorOrMatrix, randomIndex, temporaryRandomLabelVector)
		
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
	
	local trainDatasetMatrix = {}

	local validationDatasetMatrix = {}

	local testDatasetMatrix = {}
	
	for index = 1, numberOfData, 1 do
		
		if (index < numberOfTrainData) then 
			
			table.insert(trainDatasetMatrix, datasetCopy[index])
			continue
			
		end
		
		if (index < trainValidationDataMaxValue) then 

			table.insert(validationDatasetMatrix, datasetCopy[index])
			continue

		end
		
		if (index < trainValidationTestDataMaxValue) then 

			table.insert(testDatasetMatrix, datasetCopy[index])
			continue

		end
		
		if (numberOfTrainData > 0) then
			
			table.insert(trainDatasetMatrix, datasetCopy[index])
			continue
			
		end
		
		if (numberOfValidationData > 0) then

			table.insert(validationDatasetMatrix, datasetCopy[index])
			continue

		end
		
		if (numberOfTestData > 0) then

			table.insert(testDatasetMatrix, datasetCopy[index])
			continue

		end
		
	end

	trainDatasetMatrix = returnNilIfTableIsEmpty(trainDatasetMatrix)

	validationDatasetMatrix = returnNilIfTableIsEmpty(validationDatasetMatrix)

	testDatasetMatrix = returnNilIfTableIsEmpty(testDatasetMatrix)
	
	table.clear(datasetCopy)
	
	datasetCopy = nil

	return trainDatasetMatrix, validationDatasetMatrix, testDatasetMatrix
	
end

return ModelDatasetCreator
