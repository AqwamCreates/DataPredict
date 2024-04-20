local BaseModel = require("Model_BaseModel")

DivisiveHierarchicalModel = {}

DivisiveHierarchicalModel.__index = DivisiveHierarchicalModel

setmetatable(DivisiveHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = -math.huge

local defaultNumberOfcentroids = nil

local defaultDistanceFunction = "euclidean"

local defaultLinkageFunction = "minimum"

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

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)
	
	local distanceMatrix = {}
	
	for i = 1, #matrix1, 1 do
		
		distanceMatrix[i] = {}		
		
		for j = 1, #matrix2, 1 do
			
			distanceMatrix[i][j] = calculateDistance({matrix1[i]}, {matrix2[j]}, distanceFunction)
			
		end
		
	end
	
	return distanceMatrix
	
end

local function findFarthestDistance(distanceMatrix)
	
	local distance

	local maximumDistance = -math.huge

	local index1 = nil

	local index2 = nil

	for i = 1, #distanceMatrix, 1 do

		for j = 1, #distanceMatrix[1], 1 do

			distance = distanceMatrix[i][j]

			if (distance > maximumDistance) then

				maximumDistance = distance

				index1 = i

				index2 = j

			end

		end

	end

	return index1, index2
	
end

local function createNewClusters(featureMatrix, tableOfCentroids, featureMatrixIndex, distanceFunction) -- returns a table of matrices, where the individual matrices is their own clusters
	
	table.insert(tableOfCentroids, {featureMatrix[featureMatrixIndex]})
	
	table.remove(featureMatrix, featureMatrixIndex)
	
	local numberOfClusters = tableOfCentroids[1]
	
	local newTableOfCentroids = AqwamMatrixLibrary:copy(tableOfCentroids)
	
	local numberOfCentroids
	
	local featureVector
	
	local minimumDistance
	
	local centroidMatrix
	
	local centroidVector
	
	local distance
	
	local clusterIndex
	
	for d = 1, #featureMatrix, 1 do
		
		featureVector = {featureMatrix[d]}
		
		minimumDistance = math.huge
		
		for cluster = 1, numberOfClusters, 1 do
			
			centroidMatrix = tableOfCentroids[cluster]

			numberOfCentroids = #centroidMatrix

			for j = 1, numberOfCentroids, 1 do
				
				centroidVector = {centroidMatrix[j]}
				
				distance = calculateDistance(featureVector, centroidVector, distanceFunction)
				
				if (distance < minimumDistance) then
					
					minimumDistance = distance
					
					clusterIndex = cluster
					
				end

			end

		end
		
		table.insert(newTableOfCentroids[clusterIndex], centroidVector[1])
		
	end
	
	for cluster = 1, numberOfClusters, 1 do
		
		newTableOfCentroids[cluster] = AqwamMatrixLibrary:verticalMean(newTableOfCentroids[cluster])
		
	end
	
	return newTableOfCentroids
	
end

function DivisiveHierarchicalModel:setInitialcentroid(matrix)

	self.initialcentroids = matrix

end

function DivisiveHierarchicalModel.new(numberOfCentroids, distanceFunction, linkageFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)

	local NewDivisiveHierarchicalModel = BaseModel.new()

	setmetatable(NewDivisiveHierarchicalModel, DivisiveHierarchicalModel)

	NewDivisiveHierarchicalModel.highestCost = highestCost or defaultHighestCost

	NewDivisiveHierarchicalModel.lowestCost = lowestCost or defaultLowestCost

	NewDivisiveHierarchicalModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewDivisiveHierarchicalModel.linkageFunction = linkageFunction or defaultLinkageFunction

	NewDivisiveHierarchicalModel.numberOfCentroids = numberOfCentroids or defaultNumberOfcentroids

	NewDivisiveHierarchicalModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)

	NewDivisiveHierarchicalModel.initialcentroids = nil

	return NewDivisiveHierarchicalModel

end

function DivisiveHierarchicalModel:setParameters(numberOfCentroids, distanceFunction, linkageFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)

	self.highestCost = highestCost or self.highestCost

	self.lowestCost = lowestCost or self.lowestCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.linkageFunction = linkageFunction or self.linkageFunction

	self.numberOfCentroids = numberOfCentroids or self.numberOfCentroids

	self.stopWhenModelParametersDoesNotChange =  self:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)

end

function DivisiveHierarchicalModel:train(featureMatrix)

	if self.ModelParameters then

		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end

		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, self.ModelParameters)

	end

	if self.ModelParameters and self.initialcentroids then

		if (#self.initialcentroids[1] >= #self.ModelParameters[1]) then error("The number of features in initial centroids are not the same as model parameters!") end

	end

	local tableOfCentroids = AqwamMatrixLibrary:copy(featureMatrix)

	if self.initialcentroids then

		if (#self.initialcentroids >= self.numberOfcentroids) then error("The number of initial centroids is greater or equal to the number of maximum centroids!") end

		centroids = self.initialcentroid

	else
		
		local randomNumber = Random.new():NextInteger(1, #centroids)

		centroids = {centroids[randomNumber]}

	end

	local centroidIndex

	local featureMatrixIndex

	local minimumDistance

	local isOutsideCostBounds

	local numberOfIterations = 0

	local cost = 0

	local costArray = {}

	local PreviousModelParameters

	local centroidDistanceMatrix

	local centroidIndex1

	local dataToClusterDistanceMatrix

	local areModelParametersEqual = false

	repeat

		self:iterationWait()

		numberOfIterations = numberOfIterations + 1

		dataToClusterDistanceMatrix = createDistanceMatrix(featureMatrix, centroids, self.distanceFunction)
		
		featureMatrixIndex, centroidIndex = findFarthestDistance(dataToClusterDistanceMatrix)
		
		tableOfCentroids = createNewClusters(featureMatrix, tableOfCentroids, featureMatrixIndex, self.distanceFunction)
		
		

		isOutsideCostBounds = (cost <= self.lowestCost) or (cost >= self.highestCost)

		PreviousModelParameters = self.ModelParameters

	until isOutsideCostBounds or (#centroids == self.numberOfcentroids) or (numberOfIterations == #featureMatrix) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)

	self.ModelParameters = centroids

	return costArray

end

function DivisiveHierarchicalModel:predict(featureMatrix)

	local distance

	local closestcentroid

	local centroidVector

	local minimumDistance = math.huge

	for i, centroid in ipairs(self.ModelParameters) do

		centroidVector = {centroid}

		distance = calculateDistance(featureMatrix, centroidVector, self.distanceFunction)

		if (distance < minimumDistance) then

			minimumDistance = distance

			closestcentroid = i

		end

	end

	return closestcentroid, minimumDistance

end

return DivisiveHierarchicalModel
