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

local BaseModel = require("Model_BaseModel")

KNearestNeighboursRegressor = {}

KNearestNeighboursRegressor.__index = KNearestNeighboursRegressor

setmetatable(KNearestNeighboursRegressor, BaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local defaultUseWeightedDistance = false

local distanceFunctionList = {

	["Manhattan"] = function(x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		part1 = AqwamTensorLibrary:applyFunction(math.abs, part1)

		local distance = AqwamTensorLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function(x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		local part2 = AqwamTensorLibrary:power(part1, 2)

		local part3 = AqwamTensorLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,

	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamTensorLibrary:dotProduct(x1, AqwamTensorLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamTensorLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamTensorLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2, 2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

local function createDistanceMatrix(featureMatrix, storedFeatureMatrix, distanceFunction)

	local numberOfData = #featureMatrix

	local numberOfStoredData = #storedFeatureMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfStoredData}, 0)

	local calculateDistance = distanceFunctionList[distanceFunction]

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

function KNearestNeighboursRegressor.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewKNearestNeighboursRegressor = BaseModel.new(parameterDictionary)

	setmetatable(NewKNearestNeighboursRegressor, KNearestNeighboursRegressor)
	
	NewKNearestNeighboursRegressor:setName("KNearestNeighboursRegressor")

	NewKNearestNeighboursRegressor.kValue = parameterDictionary.kValue or defaultKValue

	NewKNearestNeighboursRegressor.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewKNearestNeighboursRegressor.useWeightedDistance = NewKNearestNeighboursRegressor:getValueOrDefaultValue(parameterDictionary.useWeightedDistance, defaultUseWeightedDistance)

	return NewKNearestNeighboursRegressor

end

function KNearestNeighboursRegressor:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The number of data in feature matrix and the label vector are not the same.") end

	local ModelParameters = self.ModelParameters

	if ModelParameters then

		local storedFeatureMatrix = ModelParameters[1]

		local storedLabelVector = ModelParameters[2]

		featureMatrix = AqwamTensorLibrary:concatenate(featureMatrix, storedFeatureMatrix, 1)

		labelVector = AqwamTensorLibrary:concatenate(labelVector, storedLabelVector, 1)

	end

	if (self.kValue > #featureMatrix) then warn("Number of data is less than the K value. Please add more data before doing any predictions.") end

	self.ModelParameters = {featureMatrix, labelVector}

end

function KNearestNeighboursRegressor:predict(featureMatrix, returnOriginalOutput)

	if (not self.ModelParameters) then error("No model parameters.") end

	local storedFeatureMatrix = self.ModelParameters[1]

	local storedLabelVector = self.ModelParameters[2]

	local kValue = self.kValue

	local distanceFunction = self.distanceFunction

	local useWeightedDistance = self.useWeightedDistance

	local distanceMatrix = createDistanceMatrix(featureMatrix, storedFeatureMatrix, distanceFunction)

	if (returnOriginalOutput) then return distanceMatrix end

	local predictedLabelVector = {}

	for i = 1, #featureMatrix, 1 do

		local distanceVector = {deepCopyTable(distanceMatrix[i])}

		local sortedLabelVectorLowestToHighest = deepCopyTable(storedLabelVector)

		mergeSort(distanceVector, sortedLabelVectorLowestToHighest, 1, #distanceVector[1])

		local averageValue = getAverageValue(sortedLabelVectorLowestToHighest, distanceVector, kValue, useWeightedDistance)

		predictedLabelVector[i] = {averageValue}

	end

	return predictedLabelVector

end

return KNearestNeighboursRegressor
