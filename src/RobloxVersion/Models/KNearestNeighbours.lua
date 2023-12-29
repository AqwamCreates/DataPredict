local BaseModel = require(script.Parent.BaseModel)

KNearestNeighbours = {}

KNearestNeighbours.__index = KNearestNeighbours

setmetatable(KNearestNeighbours, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local part3 = AqwamMatrixLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

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

function partition(distanceVector, storedLabelVector, low, high)
	-- Choose the pivot
	local pivot = distanceVector[1][high]

	-- Index of smaller element and indicate
	-- the right position of pivot found so far
	local i = low - 1

	for j = low, high do
		-- If current element is smaller than the pivot
		if distanceVector[1][j] < pivot then
			-- Increment index of smaller element
			i = i + 1
			distanceVector[1][i], distanceVector[1][j] = distanceVector[1][j], distanceVector[1][i]
			storedLabelVector[i][1], storedLabelVector[j][1] = storedLabelVector[j][1], storedLabelVector[i][1]
		end
	end

	distanceVector[1][i + 1], distanceVector[1][high] = distanceVector[1][high], distanceVector[1][i + 1]
	storedLabelVector[i + 1][1], storedLabelVector[high][1] = storedLabelVector[high][1], storedLabelVector[i + 1][1]

	return i + 1
end

-- The Quicksort function implementation
function quickSort(distanceVector, labelVector, low, high)
	-- When low is less than high
	if low < high then
		local partitionIndex = partition(distanceVector, labelVector, low, high)
		quickSort(distanceVector, labelVector, low, partitionIndex - 1)
		quickSort(distanceVector, labelVector, partitionIndex + 1, high)
	end
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
	
	if self.ModelParameters then
		
		local storedFeatureMatrix = self.ModelParameters[1]
		
		local storedLabelVector = self.ModelParameters[2]
		
		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, storedFeatureMatrix)
		
		labelVector = AqwamMatrixLibrary:verticalConcatenate(labelVector, storedLabelVector)
		
	end
	
	self.ModelParameters = {featureMatrix, labelVector}
	
end

function KNearestNeighbours:predict(featureMatrix, returnOriginalOutput)
	
	local storedFeatureMatrix = self.ModelParameters[1]
	
	local storedLabelVector = self.ModelParameters[2]
	
	local distanceFunction = self.distanceFunction
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, storedFeatureMatrix, distanceFunction)
	
	if returnOriginalOutput then return distanceMatrix end
	
	local predictedLabelVector = {}
	
	local numberOfDatapoints = {}
	
	for i = 1, #featureMatrix, 1 do
		
		local distanceVector = {deepCopyTable(distanceMatrix[i])}
			
		local storedLabelVectorCopy = deepCopyTable(storedLabelVector)
		
		quickSort(distanceVector, storedLabelVectorCopy, 1, #distanceVector)
		
		print(distanceVector)
			
	end
	
	return predictedLabelVector
	
end

return KNearestNeighbours
