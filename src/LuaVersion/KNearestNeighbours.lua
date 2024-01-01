local BaseModel = require("Model_BaseModel")

KNearestNeighbours = {}

KNearestNeighbours.__index = KNearestNeighbours

setmetatable(KNearestNeighbours, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local distanceFunctionList = {

	["Manhattan"] = function(x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function(x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local part3 = AqwamMatrixLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,
	
	["CosineSimilarity"] = function(x1, x2)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x2))
		
		local distancePart1 = AqwamMatrixLibrary:subtract(x1, x2)

		local distancePart2 = AqwamMatrixLibrary:power(distancePart1, 2)

		local distancePart3 = AqwamMatrixLibrary:sum(distancePart2)

		local distance = math.sqrt(distancePart3)

		local normX = AqwamMatrixLibrary:power(distance, 2)

		local kernelMatrix = AqwamMatrixLibrary:divide(dotProductedX, normX)

		return kernelMatrix

	end,

}

local function createDistanceMatrix(featureMatrix, storedFeatureMatrix, distanceFunction)

	local numberOfData = #featureMatrix

	local numberOfStoredData = #storedFeatureMatrix

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfStoredData)

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
		labelVector[indexOfMergedArray][1] = leftLabelVector[indexOfSubArrayTwo]
		indexOfSubArrayTwo = indexOfSubArrayTwo + 1
		indexOfMergedArray = indexOfMergedArray + 1
		
	end
end

local function mergeSort(distanceVector, labelVector, startingValue, endValue)
	
	if startingValue >= endValue then
		return
	end

	local mid = math.floor(startingValue + (endValue - startingValue) / 2)
	mergeSort(distanceVector, labelVector, startingValue, mid)
	mergeSort(distanceVector, labelVector, mid + 1, endValue)
	merge(distanceVector, labelVector, startingValue, mid, endValue)
	
end

local function getMajorityClass(sortedLabelVectorLowestToHighest, kValue)
	
	local classesList = {}
	
	local numberOfDataWithClassList = {}
	
	local highestNumberOfClasses = -math.huge
	
	local minimumNumberOfkValue = math.min(#sortedLabelVectorLowestToHighest, kValue)
	
	local majorityClass
	
	for k = 1, minimumNumberOfkValue, 1 do
		
		local index = table.find(classesList, sortedLabelVectorLowestToHighest[k][1])
		
		if not index then
			
			table.insert(classesList, sortedLabelVectorLowestToHighest[k][1])
			table.insert(numberOfDataWithClassList, 1)
			
			
		else
			
			numberOfDataWithClassList[index] += 1
			
		end
		
	end
	
	for index, value in ipairs(numberOfDataWithClassList) do
		
		if (value <= highestNumberOfClasses) then continue end
		
		highestNumberOfClasses = value
		
		majorityClass = classesList[index]
		
	end 
	
	return majorityClass
	
end

function KNearestNeighbours.new(kValue, distanceFunction)
	
	local newKNearestNeighbours = {}
	
	setmetatable(newKNearestNeighbours, KNearestNeighbours)
	
	newKNearestNeighbours.kValue = kValue or defaultKValue
	
	newKNearestNeighbours.distanceFunction = distanceFunction or defaultDistanceFunction
	
	return newKNearestNeighbours
	
end

function KNearestNeighbours:setParameters(kValue, distanceFunction)
	
	self.kValue = kValue or self.kValue

	self.distanceFunction = distanceFunction or self.distanceFunction
	
end

function KNearestNeighbours:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The number of data in feature matrix and the label vector are not the same!") end
	
	if self.ModelParameters then
		
		local storedFeatureMatrix = self.ModelParameters[1]
		
		local storedLabelVector = self.ModelParameters[2]
		
		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, storedFeatureMatrix)
		
		labelVector = AqwamMatrixLibrary:verticalConcatenate(labelVector, storedLabelVector)
		
	end
	
	if (self.kValue > #featureMatrix) then warn("Number of data is less than the K value. Please add more data before doing any predictions.") end
	
	self.ModelParameters = {featureMatrix, labelVector}
	
end

function KNearestNeighbours:predict(featureMatrix, returnOriginalOutput)
	
	if (#self.ModelParameters == 0) then error("No model parameters!") end
	
	local storedFeatureMatrix = self.ModelParameters[1]
	
	local storedLabelVector = self.ModelParameters[2]
	
	local distanceFunction = self.distanceFunction
	
	local kValue = self.kValue
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, storedFeatureMatrix, distanceFunction)
	
	if returnOriginalOutput then return distanceMatrix end
	
	local predictedLabelVector = {}
	
	for i = 1, #featureMatrix, 1 do
		
		local distanceVector = {deepCopyTable(distanceMatrix[i])}
			
		local sortedLabelVectorLowestToHighest = deepCopyTable(storedLabelVector)
		
		mergeSort(distanceVector, sortedLabelVectorLowestToHighest, 1, #distanceVector[1])
		
		local majorityClass = getMajorityClass(sortedLabelVectorLowestToHighest, kValue)
		
		predictedLabelVector[i] = {majorityClass}
			
	end
	
	return predictedLabelVector
	
end

return KNearestNeighbours
