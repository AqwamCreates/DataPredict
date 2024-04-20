local BaseModel = require(script.Parent.BaseModel)

DivisiveHierarchicalModel = {}

DivisiveHierarchicalModel.__index = DivisiveHierarchicalModel

setmetatable(DivisiveHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

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

local function createCentroidDistanceMatrix(featureMatrix, centroids, distanceFunction) -- m x n, where m is the data and n is the centroids

	local numberOfData = #featureMatrix

	local numberOfCentroids = #centroids

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfCentroids)

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfCentroids, 1 do

			distanceMatrix[i][j] = calculateDistance({featureMatrix[i]}, {centroids[j]} , distanceFunction)

		end

	end

	return distanceMatrix

end


local function applyFunctionToFirstRowAndColumnOfDistanceMatrix(functionToApply, newcentroidDistanceMatrix, featureMatrix, featureMatrixIndex)

	local newColumnIndex = 2

	local newRowIndex = 2

	for column = 1, #centroidDistanceMatrix, 1 do

		if (column == centroidIndex1) or (column == centroidIndex2) then continue end

		local distance = functionToApply(centroidDistanceMatrix[centroidIndex1][column],  centroidDistanceMatrix[centroidIndex2][column])

		newcentroidDistanceMatrix[1][newColumnIndex] = distance

		newColumnIndex += 1

	end

	for row = 1, #centroidDistanceMatrix, 1 do

		if (row == centroidIndex1) or (row == centroidIndex2) then continue end

		local distance = functionToApply(centroidDistanceMatrix[row][centroidIndex1],  centroidDistanceMatrix[row][centroidIndex2])

		newcentroidDistanceMatrix[newRowIndex][1] = distance

		newRowIndex += 1

	end

	return newcentroidDistanceMatrix

end

-----------------------------------------------------------------------------------------------------------------------

local function minimumLinkage(centroidDistanceMatrix)

	local newcentroidDistanceMatrix = createcentroidDistanceMatrix(featureMatrix, centroids, distanceFunction)

	newcentroidDistanceMatrix = applyFunctionToFirstRowAndColumnOfDistanceMatrix(math.min, centroidDistanceMatrix, newcentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	return newcentroidDistanceMatrix

end

local function maximumLinkage(centroidDistanceMatrix)

	local newcentroidDistanceMatrix = createcentroidDistanceMatrix(featureMatrix, centroids, distanceFunction)

	newcentroidDistanceMatrix = applyFunctionToFirstRowAndColumnOfDistanceMatrix(math.max, centroidDistanceMatrix, newcentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	return newcentroidDistanceMatrix

end

local function groupAverageLinkage(centroidDistanceMatrix)

	local weightedGroupAverage = function (x, y) return (x + y) / 2 end

	local newcentroidDistanceMatrix = createcentroidDistanceMatrix(featureMatrix, centroids, distanceFunction)

	newcentroidDistanceMatrix = applyFunctionToFirstRowAndColumnOfDistanceMatrix(weightedGroupAverage, centroidDistanceMatrix, newcentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	return newcentroidDistanceMatrix

end

local function wardLinkage(centroidDistanceMatrix)

	for i = 1, #centroidDistanceMatrix[1],1 do

		centroidDistanceMatrix[1][i] = math.pow(centroidDistanceMatrix[1][i], 2)

		centroidDistanceMatrix[2][i] = math.pow(centroidDistanceMatrix[2][i], 2)

	end

	return centroidDistanceMatrix

end

-----------------------------------------------------------------------------------------------------------------------

local function findFarthestcentroids(centroidDistanceMatrix)  -- m x n, where m is the data and n is the centroids

	local distance

	local maximumCentroidDistance = -math.huge

	local centroidIndex = nil

	local featureMatrixIndex = nil

	local numberOfData = #centroidDistanceMatrix

	local numberOfcentroids = #centroidDistanceMatrix[1]

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfcentroids, 1 do

			distance = centroidDistanceMatrix[i][j]

			if (distance > maximumCentroidDistance) then

				maximumCentroidDistance = distance

				featureMatrixIndex = i

				centroidIndex = j

			end

		end

	end

	return featureMatrixIndex, centroidIndex

end

local function updateDistanceMatrix(linkageFunction, centroidDistanceMatrix)

	local centroidVector

	local distance

	if (linkageFunction == "minimum") then

		return minimumLinkage(centroidDistanceMatrix)

	elseif (linkageFunction == "maximum") then

		return maximumLinkage(centroidDistanceMatrix)

	elseif (linkageFunction == "groupAverage") then

		return groupAverageLinkage(centroidDistanceMatrix)

	elseif (linkageFunction == "ward") then

		return wardLinkage(centroidDistanceMatrix)

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

local function splitCentroid(featureMatrix, centroids, featureMatrixIndex, centroidIndex)
	
	local featureVector = {featureMatrix[featureMatrixIndex]}

	local centroid = {centroids[centroidIndex]}

	local newCentroid = AqwamMatrixLibrary:subtract(centroid, featureVector)

	newCentroid = AqwamMatrixLibrary:multiply(newCentroid, 2)

	--table.remove(centroids, centroidIndex)

	table.insert(centroids, 1, newCentroid[1])

	--table.insert(centroids, 2, featureVector[1])

	return centroids

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

	local centroids = AqwamMatrixLibrary:copy(featureMatrix)

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

	local distance

	local areModelParametersEqual = false

	centroidDistanceMatrix = createCentroidDistanceMatrix(featureMatrix, centroids, self.distanceFunction)

	repeat

		self:iterationWait()

		numberOfIterations += 1

		featureMatrixIndex, centroidIndex = findFarthestcentroids(centroidDistanceMatrix)

		centroids = splitCentroid(featureMatrix, centroids, featureMatrixIndex, centroidIndex)

		centroidDistanceMatrix = createCentroidDistanceMatrix(featureMatrix, centroids, self.distanceFunction)

		centroidDistanceMatrix = updateDistanceMatrix(self.linkageFunction, centroidDistanceMatrix)

		self.ModelParameters = centroids

		cost = calculateCost(centroids, featureMatrix, self.distanceFunction)

		table.insert(costArray, cost)
		
		AqwamMatrixLibrary:printMatrix(centroids)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

		areModelParametersEqual = areModelParametersMatricesEqualInSizeAndValues(self.ModelParameters, PreviousModelParameters)

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
