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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

ModelDatasetCreator = {}

ModelDatasetCreator.__index = ModelDatasetCreator

setmetatable(ModelDatasetCreator, BaseInstance)

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

		if (#featureMatrix ~= #labelVectorOrMatrix) then error("The feature matrix and the label vector/matrix do not contain the same number of data.") end

	end
	
	return #featureMatrix
	
end

function ModelDatasetCreator.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewModelDatasetCreator = BaseInstance.new(parameterDictionary)

	setmetatable(NewModelDatasetCreator, ModelDatasetCreator)
	
	NewModelDatasetCreator:setName("ModelDatasetCreator")

	NewModelDatasetCreator:setClassName("ModelDatasetCreator")
	
	NewModelDatasetCreator.trainDataRatio = parameterDictionary.trainDataRatio or defaultTrainDataRatio
	
	NewModelDatasetCreator.validationDataRatio = parameterDictionary.validationDataRatio or defaultValidationDataRatio
	
	NewModelDatasetCreator.testDataRatio = parameterDictionary.testDataRatio or defaultTestDataRatio
	
	NewModelDatasetCreator.datasetRandomizationProbability = parameterDictionary.datasetRandomizationProbability or defaultDatasetRandomizationProbability
	
	return NewModelDatasetCreator
	
end

function ModelDatasetCreator:setDatasetSplitPercentages(trainDataRatio, validationDataRatio, testDataPercentage)
	
	self.trainDataRatio = trainDataRatio or self.trainDataRatio

	self.validationDataRatio = validationDataRatio or self.validationDataRatio

	self.testDataRatio = testDataPercentage or self.testDataRatio
	
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
	
	local numberOfTrainData = math.floor(self.trainDataRatio * numberOfData)
	
	local numberOfValidationData = math.floor(self.validationDataRatio * numberOfData)
	
	local numberOfTestData = math.floor(self.testDataRatio * numberOfData)
	
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
