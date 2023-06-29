local BaseModel = require(script.Parent.BaseModel)

DivisiveHierarchicalModel = {}

DivisiveHierarchicalModel.__index = DivisiveHierarchicalModel

setmetatable(DivisiveHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = -math.huge

local defaultNumberOfClusters = nil

local defaultDistanceFunction = "euclidean"

local defaultStopWhenModelParametersDoesNotChange = false

local distanceFunctionList = {
	
	["manhattan"] = function (x1, x2)
		
		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)
		
		local distance = AqwamMatrixLibrary:sum(part1)
		
		return distance 
		
	end,
	
	["euclidean"] = function (x1, x2)
		
		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		local part2 = AqwamMatrixLibrary:power(part1, 2)
		
		local part3 = AqwamMatrixLibrary:sum(part2)
		
		local distance = math.sqrt(part3)
		
		return distance 
		
	end,
}

local function calculateDistance(vector1, vector2, distanceFunction)
	
	return distanceFunctionList[distanceFunction](vector1, vector2) 
	
end

local function createClusterDistanceMatrix(featureMatrix, clusters, distanceFunction) -- m x n, where m is the data and n is the clusters

	local numberOfData = #featureMatrix
	
	local numberOfClusters = #clusters

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfClusters, 1 do

			distanceMatrix[i][j] = calculateDistance({featureMatrix[i]}, {clusters[j]} , distanceFunction)

		end

	end

	return distanceMatrix

end

local function createNewMergedDistanceMatrix(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local numberOfData = #clusterDistanceMatrix

	local newClusterDistanceMatrix = {}

	for i = 1, numberOfData, 1 do

		if (i == clusterIndex1) or (i == clusterIndex2) then continue end

		local newClusterDistanceVector = {}

		for j = 1, numberOfData, 1 do

			if (j == clusterIndex1) or (j == clusterIndex2) then continue end

			table.insert(newClusterDistanceVector, clusterDistanceMatrix[i][j])

		end

		table.insert(newClusterDistanceMatrix, newClusterDistanceVector)

	end

	local newRow = {}

	for i = 1, (numberOfData - 2) do

		table.insert(newRow, 0)

	end

	table.insert(newClusterDistanceMatrix, 1, newRow)

	for i = 1, (numberOfData - 1) do

		table.insert(newClusterDistanceMatrix[i], 1, 0)

	end

	return newClusterDistanceMatrix

end

local function applyFunctionToFirstRowAndColumnOfDistanceMatrix(functionToApply, clusterDistanceMatrix, newClusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local totalDistance = 0

	local newColumnIndex = 2

	local newRowIndex = 2

	for column = 1, #clusterDistanceMatrix, 1 do

		if (column == clusterIndex1) or (column == clusterIndex2) then continue end

		local distance = functionToApply(clusterDistanceMatrix[clusterIndex1][column],  clusterDistanceMatrix[clusterIndex2][column])

		newClusterDistanceMatrix[1][newColumnIndex] = distance

		newColumnIndex += 1

	end

	for row = 1, #clusterDistanceMatrix, 1 do

		if (row == clusterIndex1) or (row == clusterIndex2) then continue end

		local distance = functionToApply(clusterDistanceMatrix[row][clusterIndex1],  clusterDistanceMatrix[row][clusterIndex2])

		newClusterDistanceMatrix[newRowIndex][1] = distance

		totalDistance += distance

		newRowIndex += 1

	end

	return newClusterDistanceMatrix, totalDistance

end

-----------------------------------------------------------------------------------------------------------------------

local function minimumLinkage(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local newClusterDistanceMatrix = createNewMergedDistanceMatrix(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local newClusterDistanceMatrix, totalDistance = applyFunctionToFirstRowAndColumnOfDistanceMatrix(math.min, clusterDistanceMatrix, newClusterDistanceMatrix, clusterIndex1, clusterIndex2)

	return newClusterDistanceMatrix, totalDistance

end

local function maximumLinkage(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local newClusterDistanceMatrix = createNewMergedDistanceMatrix(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local newClusterDistanceMatrix, totalDistance = applyFunctionToFirstRowAndColumnOfDistanceMatrix(math.max, clusterDistanceMatrix, newClusterDistanceMatrix, clusterIndex1, clusterIndex2)

	return newClusterDistanceMatrix, totalDistance

end

local function groupAverageLinkage(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local weightedGroupAverage = function (x, y) return (x + y) / 2 end

	local newClusterDistanceMatrix = createNewMergedDistanceMatrix(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local newClusterDistanceMatrix, totalDistance = applyFunctionToFirstRowAndColumnOfDistanceMatrix(weightedGroupAverage, clusterDistanceMatrix, newClusterDistanceMatrix, clusterIndex1, clusterIndex2)

	return newClusterDistanceMatrix, totalDistance

end

local function wardLinkage(clusters, clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local newClusterDistanceMatrix = createClusterDistanceMatrix(clusters, "euclidean")

	for i = 2, #newClusterDistanceMatrix,1 do

		newClusterDistanceMatrix[1][i] = math.pow(newClusterDistanceMatrix[1][i], 2)

		newClusterDistanceMatrix[i][1] = math.pow(newClusterDistanceMatrix[i][1], 2)

	end

	return newClusterDistanceMatrix

end

-----------------------------------------------------------------------------------------------------------------------

local function findFarthestClusters(clusterDistanceMatrix)  -- m x n, where m is the data and n is the clusters
	
	local distance

	local maximumClusterDistance = -math.huge

	local featureMatrixIndex = nil
	
	local numberOfData = #clusterDistanceMatrix
	
	local numberOfClusters = #clusterDistanceMatrix[1]
	
	local totalDistanceMatrixFromClusters = AqwamMatrixLibrary:horizontalSum(clusterDistanceMatrix)

	for i = 1, #numberOfData, 1 do
		
		for j = 1, #numberOfData, 1 do
			
			distance = totalDistanceMatrixFromClusters[i][j]
			
			if (distance > maximumClusterDistance) then

				maximumClusterDistance = distance

				featureMatrixIndex = i

			end
			
		end
		
	end

	return featureMatrixIndex
	
end

local function updateDistanceMatrix(linkageFunction, clusters, clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	local clusterVector

	local distance

	if (linkageFunction == "minimum") then

		return minimumLinkage(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	elseif (linkageFunction == "maximum") then

		return maximumLinkage(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	elseif (linkageFunction == "groupAverage") then

		return groupAverageLinkage(clusterDistanceMatrix, clusterIndex1, clusterIndex2)

	else

		error("Invalid linkage!")

	end

end

local function findEqualRowIndex(matrix1, matrix2)

	local index

	for i = 1, #matrix1, 1 do

		local matrixInTable = {matrix1[i]}

		if AqwamMatrixLibrary:areMatricesEqual(matrixInTable, matrix2) then

			index = i

			break

		end

	end

	return index

end

local function removeDuplicateRows(ModelParameters)

	local UniqueModelParameters = {}

	for i = 1, #ModelParameters, 1 do

		local index = findEqualRowIndex(UniqueModelParameters, {ModelParameters[i]})

		if (index == nil) then table.insert(UniqueModelParameters, ModelParameters[i]) end

	end

	return UniqueModelParameters

end

local function splitCluster(cluster, centroid1, centroid2, distanceFunction)
		
	local cluster1 = {}
		
	local cluster2 = {}

	for i = 3, #cluster do
			
		local point = cluster[i]
			
		local distance1 = calculateDistance(point, centroid1, distanceFunction)
			
		local distance2 = calculateDistance(point, centroid2, distanceFunction)

		if distance1 < distance2 then
				
			table.insert(cluster1, point)
				
			centroid1[1] = (centroid1[1] + point[1]) / 2
				
			centroid1[2] = (centroid1[2] + point[2]) / 2
				
		else
				
			table.insert(cluster2, point)
				
			centroid2[1] = (centroid2[1] + point[1]) / 2
				
			centroid2[2] = (centroid2[2] + point[2]) / 2
				
		end
			
	end

	return cluster1, cluster2
		
end

local function calculateCost(centroids, featureMatrix, distanceFunction)

	local cost = 0

	for i = 1, #featureMatrix, 1 do

		local featureVector = {featureMatrix[i]}

		local minimumDistance = math.huge

		for j = 1, #centroids, 1 do

			local centroid = {centroids[j]}

			local distance = calculateDistance(featureVector, centroid, distanceFunction)

			minimumDistance = math.min(minimumDistance, distance)

		end

		cost += minimumDistance

	end

	return cost

end

local function areModelParametersMatricesEqualInSizeAndValues(ModelParameters, PreviousModelParameters)
	
	local areModelParametersEqual = false
	
	if (PreviousModelParameters == nil) then return areModelParametersEqual end
	
	if (#ModelParameters ~= #PreviousModelParameters) or (#ModelParameters[1] ~= #PreviousModelParameters[1]) then return areModelParametersEqual end
	
	areModelParametersEqual = AqwamMatrixLibrary:areMatricesEqual(ModelParameters, PreviousModelParameters)
	
	return areModelParametersEqual
	
end

function DivisiveHierarchicalModel:setInitialCluster(matrix)
	
	self.initialClusters = matrix
	
end

function DivisiveHierarchicalModel.new(numberOfClusters, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	local NewDivisiveHierarchicalModel = BaseModel.new()
	
	setmetatable(NewDivisiveHierarchicalModel, DivisiveHierarchicalModel)
	
	NewDivisiveHierarchicalModel.highestCost = highestCost or defaultHighestCost
	
	NewDivisiveHierarchicalModel.lowestCost = lowestCost or defaultLowestCost
	
	NewDivisiveHierarchicalModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	NewDivisiveHierarchicalModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters
	
	NewDivisiveHierarchicalModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)
	
	NewDivisiveHierarchicalModel.initialClusters = nil
	
	return NewDivisiveHierarchicalModel
	
end

function DivisiveHierarchicalModel:setParameters(numberOfClusters, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	self.highestCost = highestCost or self.highestCost
	
	self.lowestCost = lowestCost or self.lowestCost
	
	self.distanceFunction = distanceFunction or self.distanceFunction
	
	self.numberOfClusters = numberOfClusters or self.numberOfClusters
	
	self.stopWhenModelParametersDoesNotChange =  self:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)
	
end

function DivisiveHierarchicalModel:train(featureMatrix)
	
	if self.ModelParameters then

		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end

		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, self.ModelParameters)

	end

	if self.ModelParameters and self.initialClusters then

		if (#self.initialClusters[1] >= #self.ModelParameters[1]) then error("The number of features in initial clusters are not the same as model parameters!") end

	end
	
	local clusters = AqwamMatrixLibrary:copy(featureMatrix)
	
	if self.initialClusters then

		if (#self.initialClusters >= self.numberOfClusters) then error("The number of initial clusters is greater or equal to the number of maximum clusters!") end

		clusters = self.initialCluster

	else

		clusters = {clusters[Random.new():NextInteger(1, #clusters)]}

	end
	
	local clusterIndex

	local featureMatrixIndex1
	
	local featureMatrixIndex2

	local minimumDistance

	local isOutsideCostBounds

	local numberOfIterations = 0

	local cost = 0

	local costArray = {}

	local PreviousModelParameters

	local clusterDistanceMatrix

	local clusterIndex1

	local distance

	local areModelParametersEqual = false

	local newCluster
	
	clusterDistanceMatrix = createClusterDistanceMatrix(featureMatrix, clusters, self.distanceFunction)

	repeat

		self:iterationWait()

		numberOfIterations += 1

		clusterIndex, featureMatrixIndex1, featureMatrixIndex2 = findFarthestClusters(clusterDistanceMatrix, featureMatrix)

		clusters = createNewClusters(featureMatrix, featureMatrixIndex1, featureMatrixIndex2)

		clusterDistanceMatrix, distance = updateDistanceMatrix(self.linkageFunction, clusters, clusterDistanceMatrix, clusterIndex, featureMatrixIndex1)

		self.ModelParameters = clusters

		cost = distance

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

		areModelParametersEqual = areModelParametersMatricesEqualInSizeAndValues(self.ModelParameters, PreviousModelParameters)

		isOutsideCostBounds = (cost <= self.lowestCost) or (cost >= self.highestCost)

		PreviousModelParameters = self.ModelParameters

	until isOutsideCostBounds or (#clusters == self.numberOfClusters) or (#clusters == 1) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)

	self.ModelParameters = clusters

	return costArray
	
end

function DivisiveHierarchicalModel:predict(featureMatrix)
	
	local distance
	
	local closestCluster
	
	local clusterVector
	
	local minimumDistance = math.huge
	
	for i, cluster in ipairs(self.ModelParameters) do
		
		clusterVector = {cluster}
		
		distance = calculateDistance(featureMatrix, clusterVector, self.distanceFunction)
		
		if (distance < minimumDistance) then
			
			minimumDistance = distance
			
			closestCluster = i
			
		end
		
	end
	
	return closestCluster, minimumDistance
	
end

return DivisiveHierarchicalModel
