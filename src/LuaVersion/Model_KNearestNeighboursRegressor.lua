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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseModel = require("Model_BaseModel")

local distanceFunctionDictionary = require("Core_DistanceFunctionDictionary")

KNearestNeighboursRegressorModel = {}

KNearestNeighboursRegressorModel.__index = KNearestNeighboursRegressorModel

setmetatable(KNearestNeighboursRegressorModel, BaseModel)

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local defaultUseWeightedDistance = false

local defaultMaximumNumberOfData = math.huge

local function createDistanceMatrix(distanceFunction, featureMatrix, storedFeatureMatrix)

	local numberOfData = #featureMatrix

	local numberOfStoredData = #storedFeatureMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfStoredData}, 0)

	local calculateDistance = distanceFunctionDictionary[distanceFunction]

	for datasetIndex = 1, numberOfData, 1 do

		for storedDatasetIndex = 1, numberOfStoredData, 1 do

			distanceMatrix[datasetIndex][storedDatasetIndex] = calculateDistance({featureMatrix[datasetIndex]}, {storedFeatureMatrix[storedDatasetIndex]})

		end

	end

	return distanceMatrix

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

local function merge(distanceVector, labelVector, left, mid, right)

	local subArrayOne = mid - left + 1
	local subArrayTwo = right - mid

	local leftDistanceVector = {}
	local rightDistanceVector = {}

	local leftLabelVector = {}
	local rightLabelVector = {}

	for i = 1, subArrayOne do

		leftDistanceVector[i] = distanceVector[1][left + i - 1]
		leftLabelVector[i] = labelVector[left + i - 1][1]

	end

	for j = 1, subArrayTwo do

		rightDistanceVector[j] = distanceVector[1][mid + j]
		rightLabelVector[j] = labelVector[mid + j][1]

	end

	local indexOfSubArrayOne = 1
	local indexOfSubArrayTwo = 1
	local indexOfMergedArray = left

	while indexOfSubArrayOne <= subArrayOne and indexOfSubArrayTwo <= subArrayTwo do

		if leftDistanceVector[indexOfSubArrayOne] <= rightDistanceVector[indexOfSubArrayTwo] then

			distanceVector[1][indexOfMergedArray] = leftDistanceVector[indexOfSubArrayOne]
			labelVector[indexOfMergedArray][1] = leftLabelVector[indexOfSubArrayOne]
			indexOfSubArrayOne = indexOfSubArrayOne + 1

		else

			distanceVector[1][indexOfMergedArray] = rightDistanceVector[indexOfSubArrayTwo]
			labelVector[indexOfMergedArray][1] = rightLabelVector[indexOfSubArrayTwo]
			indexOfSubArrayTwo = indexOfSubArrayTwo + 1

		end

		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayOne <= subArrayOne) do

		distanceVector[1][indexOfMergedArray] = leftDistanceVector[indexOfSubArrayOne]
		labelVector[indexOfMergedArray][1] = leftLabelVector[indexOfSubArrayOne]
		indexOfSubArrayOne = indexOfSubArrayOne + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayTwo <= subArrayTwo) do

		distanceVector[1][indexOfMergedArray] = rightDistanceVector[indexOfSubArrayTwo]
		labelVector[indexOfMergedArray][1] = rightLabelVector[indexOfSubArrayTwo]
		indexOfSubArrayTwo = indexOfSubArrayTwo + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

end

local function mergeSort(distanceVector, labelVector, startingValue, endValue)

	if (startingValue >= endValue) then return end

	local mid = math.floor(startingValue + (endValue - startingValue) / 2)

	mergeSort(distanceVector, labelVector, startingValue, mid)
	mergeSort(distanceVector, labelVector, mid + 1, endValue)
	merge(distanceVector, labelVector, startingValue, mid, endValue)

end

local function getAverageValue(sortedLabelVectorLowestToHighest, distanceVector, kValue, useWeightedDistance)

	local sum = 0

	local totalWeight = 0

	local minimumNumberOfkValue = math.min(#sortedLabelVectorLowestToHighest, kValue)

	for k = 1, minimumNumberOfkValue, 1 do

		local label = sortedLabelVectorLowestToHighest[k][1]

		local distance = distanceVector[1][k]

		local weight

		if (useWeightedDistance) then

			weight = ((distance == 0) and math.huge) or (1 / distance)

		else

			weight = 1

		end

		sum = sum + (label * weight)
		
		totalWeight = totalWeight + weight

	end

	local averageValue = sum / totalWeight

	return averageValue

end

function KNearestNeighboursRegressorModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewKNearestNeighboursRegressorModel = BaseModel.new(parameterDictionary)

	setmetatable(NewKNearestNeighboursRegressorModel, KNearestNeighboursRegressorModel)
	
	NewKNearestNeighboursRegressorModel:setName("KNearestNeighboursRegressor")

	NewKNearestNeighboursRegressorModel.kValue = parameterDictionary.kValue or defaultKValue

	NewKNearestNeighboursRegressorModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewKNearestNeighboursRegressorModel.useWeightedDistance = NewKNearestNeighboursRegressorModel:getValueOrDefaultValue(parameterDictionary.useWeightedDistance, defaultUseWeightedDistance)
	
	NewKNearestNeighboursRegressorModel.maximumNumberOfData = parameterDictionary.maximumNumberOfData or defaultMaximumNumberOfData
	
	return NewKNearestNeighboursRegressorModel

end

function KNearestNeighboursRegressorModel:train(featureMatrix, labelVector)

	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The number of data in feature matrix and the label vector are not the same.") end

	local maximumNumberOfData = self.maximumNumberOfData

	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		local storedFeatureMatrix = ModelParameters[1]

		local storedLabelVector = ModelParameters[2]

		featureMatrix = AqwamTensorLibrary:concatenate(featureMatrix, storedFeatureMatrix, 1)

		labelVector = AqwamTensorLibrary:concatenate(labelVector, storedLabelVector, 1)

		numberOfData = #featureMatrix

		if (numberOfData > maximumNumberOfData) then

			local newFeatureMatrix = {}

			local newLabelVector = {}

			local dataShiftIndex = (numberOfData - maximumNumberOfData)

			for dataIndex = 1, maximumNumberOfData, 1 do

				newFeatureMatrix[dataIndex] = featureMatrix[dataIndex + dataShiftIndex]

				newLabelVector[dataIndex] = labelVector[dataIndex + dataShiftIndex]

			end

			featureMatrix = newFeatureMatrix

			labelVector = newLabelVector

			numberOfData = maximumNumberOfData

		end

	end

	if (self.kValue > numberOfData) then warn("Number of data is less than the K value. Please add more data before doing any predictions.") end

	self.ModelParameters = {featureMatrix, labelVector}

end

function KNearestNeighboursRegressorModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then return AqwamTensorLibrary:createTensor({#featureMatrix, 1}, math.huge) end

	local storedFeatureMatrix = ModelParameters[1]

	local storedLabelVector = ModelParameters[2]

	local kValue = self.kValue

	local distanceFunction = self.distanceFunction

	local useWeightedDistance = self.useWeightedDistance

	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix, storedFeatureMatrix)

	if (returnOriginalOutput) then return distanceMatrix end
	
	local numberOfOtherData = #storedFeatureMatrix

	local predictedLabelVector = {}

	for i, unwrappedDistanceVector in ipairs(distanceMatrix) do

		local distanceVector = {deepCopyTable(unwrappedDistanceVector)}

		local sortedLabelVectorLowestToHighest = deepCopyTable(storedLabelVector)

		mergeSort(distanceVector, sortedLabelVectorLowestToHighest, 1, numberOfOtherData)

		local averageValue = getAverageValue(sortedLabelVectorLowestToHighest, distanceVector, kValue, useWeightedDistance)

		predictedLabelVector[i] = {averageValue}

	end

	return predictedLabelVector

end

return KNearestNeighboursRegressorModel
