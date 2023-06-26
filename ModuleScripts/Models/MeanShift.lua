local BaseModel = require(script.Parent.BaseModel)

MeanShiftModel = {}

MeanShiftModel.__index = MeanShiftModel

setmetatable(MeanShiftModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = -math.huge

local defaultDistanceFunction = "euclidean"

local defaultStopWhenModelParametersDoesNotChange = true

local defaultBandwidth = 5

local defaultBandwidthStep = 100

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

local function assignToCluster(distanceFromClusterRowVector) -- Number of columns -> number of clusters
	
	local distanceFromCluster
	
	local shortestDistance = math.huge
	
	local clusterNumber
	
	for cluster = 1, #distanceFromClusterRowVector[1], 1 do
		
		distanceFromCluster = distanceFromClusterRowVector[1][cluster]
		
		if (distanceFromCluster < shortestDistance) then
			
			shortestDistance = distanceFromCluster
			
			clusterNumber = cluster
			
		end
		
	end
	
	return clusterNumber, shortestDistance
	
end

local function checkIfTheDataPointClusterNumberBelongsToTheCluster(dataPointClusterNumber, cluster)
	
	if (dataPointClusterNumber == cluster) then
		
		return 1
		
	else
		
		return 0
		
	end
	
end

local function createDistanceMatrix(modelParameters, featureMatrix, distanceFunction)

	local numberOfData = #featureMatrix

	local numberOfClusters = #modelParameters

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfClusters)

	for datasetIndex = 1, #featureMatrix, 1 do

		for cluster = 1, #modelParameters, 1 do

			distanceMatrix[datasetIndex][cluster] = calculateDistance({featureMatrix[datasetIndex]}, {modelParameters[cluster]} , distanceFunction)

		end

	end

	return distanceMatrix

end

local function calculateCost(modelParameters, featureMatrix, distanceFunction)
	
	local cost = 0
	
	for i = 1, #featureMatrix do
		
		local minDistance = math.huge
		
		for j = 1, #modelParameters do
			
			local distance = calculateDistance({featureMatrix[i]}, {modelParameters[j]}, distanceFunction)
		
			minDistance = math.min(minDistance, distance)
			
		end
		
		cost = cost + minDistance
		
	end
	
	return cost
	
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

local function mergeCentroids(ModelParameters, featureMatrix, bandwidth, weights, distanceFunction)
	
	local bandwidthStep = #weights
	
	local NewModelParameters = {}
	
	for i = 1, #ModelParameters, 1 do
		
		local inBandwidth = {}
		
		local centroid = {ModelParameters[i]}
		
		for j = 1, #featureMatrix, 1 do
			
			local featureVector = {featureMatrix[j]}
			
			local distance = calculateDistance(featureVector, centroid, distanceFunction)
			
			if (bandwidthStep <= 0) then
				
				if (distance <= bandwidth) then table.insert(inBandwidth, featureVector[1]) end
				
			else
				
				if (distance == 0) then distance = 0.00000001 end
				
				local weightIndex = math.floor(distance/bandwidth)
				
				if (weightIndex > (bandwidthStep - 1)) then weightIndex = (bandwidthStep - 1) end
				
				local multiplyFactor = math.pow(weights[weightIndex], 2)
				
				local newCentroid = AqwamMatrixLibrary:multiply(featureVector, multiplyFactor)
				
				table.insert(inBandwidth, newCentroid[1]) 
				
			end
			
		end
		
		if (#inBandwidth > 0) then
			
			local verticalSum = AqwamMatrixLibrary:verticalSum(inBandwidth)

			local numberOfData = #inBandwidth

			local newCentroid = AqwamMatrixLibrary:divide(verticalSum, numberOfData)

			table.insert(NewModelParameters, newCentroid[1])
			
		end
		
	end
	
	NewModelParameters = removeDuplicateRows(NewModelParameters)
	
	return NewModelParameters
	
end

function MeanShiftModel.new(maxNumberOfIterations, bandwidth, bandwidthStep, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	local NewMeanShiftModel = BaseModel.new()
	
	setmetatable(NewMeanShiftModel, MeanShiftModel)
	
	NewMeanShiftModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewMeanShiftModel.highestCost = highestCost or defaultHighestCost

	NewMeanShiftModel.lowestCost = lowestCost or defaultLowestCost

	NewMeanShiftModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewMeanShiftModel.bandwidth = bandwidth or defaultBandwidth
	
	NewMeanShiftModel.bandwidthStep = bandwidthStep or defaultBandwidthStep
	
	NewMeanShiftModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)
	
	return NewMeanShiftModel
	
end

function MeanShiftModel:setParameters(maxNumberOfIterations, bandwidth, bandwidthStep, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.highestCost = highestCost or self.highestCost

	self.lowestCost = lowestCost or self.lowestCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.bandwidth = bandwidth or self.bandwidth
	
	self.bandwidthStep = bandwidthStep or self.bandwidthStep

	self.stopWhenModelParametersDoesNotChange =  self:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)
	
end

local function checkIfModelParametersAreEqual(ModelParameters, PreviousModelParameters)
	
	local areEqual = true
	
	for i = 1, #ModelParameters, 1 do
		
		local centroid = {ModelParameters[i]}
		
		for j = 1, #PreviousModelParameters, 1 do
			
			local previousCentroid = {PreviousModelParameters[j]}
			
			areEqual = AqwamMatrixLibrary:areMatricesEqual(centroid, previousCentroid)
			
			if (areEqual == false) then break end
			
		end
		
		if (areEqual == false) then break end
		
	end
	
	return areEqual
	
end

function MeanShiftModel:train(featureMatrix)
	
	local isOutsideCostBounds
	
	local PreviousModelParameters
	
	local areModelParametersEqual
	
	local cost
	
	local costArray = {}
	
	local weights = {}
	
	local numberOfIterations = 0
	
	if (self.ModelParameters) then
		
		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		self.ModelParameters = featureMatrix
		
	end
	
	if (self.bandwidth <= 0) then
		
		local verticalMean = AqwamMatrixLibrary:verticalMean(featureMatrix)
		
		local zeroMatrix = AqwamMatrixLibrary:createMatrix(1, #featureMatrix[1])
		
		local distance = calculateDistance(verticalMean, zeroMatrix, self.distanceFunction)
		
		self.bandwidth = distance / self.bandwidthStep
		
		for i = 1, self.bandwidthStep, 1 do table.insert(weights, i) end

	end
	
	repeat
		
		self:iterationWait()

		numberOfIterations += 1
		
		PreviousModelParameters = self.ModelParameters

		self.ModelParameters = mergeCentroids(self.ModelParameters, featureMatrix, self.bandwidth, weights, self.distanceFunction)

		areModelParametersEqual = checkIfModelParametersAreEqual(self.ModelParameters, PreviousModelParameters)
		
		cost = calculateCost(self.ModelParameters, featureMatrix, self.distanceFunction)
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
		isOutsideCostBounds = (cost <= self.lowestCost) or (cost >= self.highestCost)

	until (numberOfIterations == self.maxNumberOfIterations) or isOutsideCostBounds or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	return costArray
	
end

function MeanShiftModel:predict(featureMatrix)
	
	local distanceFromClusterRowVector = createDistanceMatrix(self.ModelParameters, featureMatrix, self.distanceFunction)

	local clusterNumber, shortestDistance = assignToCluster(distanceFromClusterRowVector)
	
	return clusterNumber, shortestDistance
	
end

return MeanShiftModel
