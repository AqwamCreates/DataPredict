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

local BaseInstance = require("Core_BaseInstance")

DatasetCreator = {}

DatasetCreator.__index = DatasetCreator

setmetatable(DatasetCreator, BaseInstance)

local defaultTrainDataRatio = 0.7

local defaultValidationDataRatio = 0

local defaultTestDataRatio = 0.3

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

		if (#featureMatrix ~= #labelVectorOrMatrix) then error("The feature matrix and the label vector / matrix do not contain the same number of data.") end

	end
	
	return #featureMatrix
	
end

function DatasetCreator.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDatasetCreator = BaseInstance.new(parameterDictionary)

	setmetatable(NewDatasetCreator, DatasetCreator)
	
	NewDatasetCreator:setName("DatasetCreator")

	NewDatasetCreator:setClassName("DatasetCreator")
	
	NewDatasetCreator.trainDataRatio = parameterDictionary.trainDataRatio or defaultTrainDataRatio
	
	NewDatasetCreator.validationDataRatio = parameterDictionary.validationDataRatio or defaultValidationDataRatio
	
	NewDatasetCreator.testDataRatio = parameterDictionary.testDataRatio or defaultTestDataRatio
	
	NewDatasetCreator.datasetRandomizationProbability = parameterDictionary.datasetRandomizationProbability or defaultDatasetRandomizationProbability
	
	return NewDatasetCreator
	
end

function DatasetCreator:setDatasetSplitPercentages(trainDataRatio, validationDataRatio, testDataRatio)
	
	trainDataRatio = trainDataRatio or self.trainDataRatio
	
	validationDataRatio = validationDataRatio or self.validationDataRatio
	
	testDataRatio = testDataRatio or self.testDataRatio
	
	local ratioSum = trainDataRatio + validationDataRatio + testDataRatio
	
	if (ratioSum > 1) then error("The sum of the ratios exceeds 1.") end
	
	self.trainDataRatio = trainDataRatio

	self.validationDataRatio = validationDataRatio

	self.testDataRatio = testDataRatio
	
end

function DatasetCreator:setDatasetRandomizationProbability(datasetRandomizationProbability)
	
	self.datasetRandomizationProbability = datasetRandomizationProbability or self.datasetRandomizationProbability
	
end

function DatasetCreator:randomizeDataset(featureMatrix, labelVectorOrMatrix)
	
	local numberOfData = checkNumberOfData(featureMatrix, labelVectorOrMatrix)
	
	local datasetRandomizationProbability = self.datasetRandomizationProbability
	
	local randomizedFeatureMatrix = deepCopyTable(featureMatrix)
	
	local randomizedLabelVectorOrMatrix = deepCopyTable(labelVectorOrMatrix)
	
	for index = 1, numberOfData, 1 do
		
		if (datasetRandomizationProbability < math.random()) then
			
			local randomIndex = math.random(0, index)

			randomIndex = math.ceil(randomIndex)

			local temporaryRandomFeatureVector = randomizedFeatureMatrix[index]

			table.remove(randomizedFeatureMatrix, index)

			table.insert(randomizedFeatureMatrix, randomIndex, temporaryRandomFeatureVector)

			if (type(labelVectorOrMatrix) == "nil") then 
				
				local temporaryRandomLabelVector = randomizedLabelVectorOrMatrix[index]

				table.remove(randomizedLabelVectorOrMatrix, index)

				table.insert(randomizedLabelVectorOrMatrix, randomIndex, temporaryRandomLabelVector)
				
			end
			
		end
		
	end
	
	return randomizedFeatureMatrix, randomizedLabelVectorOrMatrix
	
end

function DatasetCreator:splitDataset(datasetMatrix)

	local numberOfData = checkNumberOfData(datasetMatrix)
	
	local datasetCopy = deepCopyTable(datasetMatrix)

	local numberOfTrainData = math.floor(self.trainDataRatio * numberOfData)
	
	local numberOfValidationData = math.floor(self.validationDataRatio * numberOfData)
	
	local numberOfTestData = math.floor(self.testDataRatio * numberOfData)

	local trainDataMaximumValue = numberOfTrainData
	
	local trainValidationDataMaximumValue = numberOfTrainData + numberOfValidationData
	
	local trainValidationTestDataMaximumValue = trainValidationDataMaximumValue + numberOfTestData

	local trainDatasetMatrix = {}
	
	local validationDatasetMatrix = {}
	
	local testDatasetMatrix = {}

	for index = 1, numberOfData, 1 do

		if (index < trainDataMaximumValue) then 

			table.insert(trainDatasetMatrix, datasetCopy[index])

		elseif (index < trainValidationDataMaximumValue) then

			table.insert(validationDatasetMatrix, datasetCopy[index])

		elseif (index < trainValidationTestDataMaximumValue) then

			table.insert(testDatasetMatrix, datasetCopy[index])

		elseif (numberOfTrainData > 0) then

			table.insert(trainDatasetMatrix, datasetCopy[index])

		elseif (numberOfValidationData > 0) then

			table.insert(validationDatasetMatrix, datasetCopy[index])

		elseif (numberOfTestData > 0) then

			table.insert(testDatasetMatrix, datasetCopy[index])

		end

	end

	trainDatasetMatrix = returnNilIfTableIsEmpty(trainDatasetMatrix)
	
	validationDatasetMatrix = returnNilIfTableIsEmpty(validationDatasetMatrix)
	
	testDatasetMatrix = returnNilIfTableIsEmpty(testDatasetMatrix)

	table.clear(datasetCopy)
	
	datasetCopy = nil

	return trainDatasetMatrix, validationDatasetMatrix, testDatasetMatrix
	
end

return DatasetCreator
