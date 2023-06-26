local BaseModel = require(script.Parent.BaseModel)

DivisiveHierarchicalModel = {}

DivisiveHierarchicalModel.__index = DivisiveHierarchicalModel

setmetatable(DivisiveHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = -math.huge

local defaultNumberOfClusters = nil

local defaultDistanceFunction = "manhattan"

local defaultLinkageMethod = "ward"

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

local linkageMethodList = {
	
	["minimum"] = function (distance1, distance2)
		
		return math.min(distance1, distance2)
		
	end,
	
	["maximum"] = function (distance1, distance2)
		
		return math.max(distance1, distance2)
		
	end,
	
	["groupAverage"] = function (distance1, distance2)
		
		return (distance1 + distance2) / 2
		
	end,
	
	["ward"] = function (distance1, distance2, size1, size2, sizeMerged)
		
		local part1 = 3 * distance1
		
		local part2 = 3 * distance2
		
		local part3 = 2 * ((distance1 + distance2) / 2)
		
		return math.sqrt((part1 + part2 + part3) / 4)
		
	end,
}

local function calculateDistance(vector1, vector2, distanceFunction)
	
	return distanceFunctionList[distanceFunction](vector1, vector2) 
	
end

local function findFarthestCluster(clusters, distanceFunction, linkageMethod)
	
	local clusterIndex1 = 1

	local clusterIndex2 = 1
	
	local maxDistance = calculateDistance({clusters[clusterIndex2]}, {clusters[clusterIndex2]}, distanceFunction)
	
	for i = 1, #clusters do
		
		for j = i + 1, #clusters do
			
			local distance = calculateDistance({clusters[i]}, {clusters[j]}, distanceFunction)
			
			local linkageDistance = linkageMethodList[linkageMethod](distance, maxDistance)
			
			if (linkageDistance > maxDistance) then
				
				maxDistance = distance
				
				clusterIndex1 = i
				
				clusterIndex2 = j
				
			end
			
		end
		
	end
	
	return clusterIndex1, clusterIndex2, maxDistance
	
end

local function splitCluster(clusters, clusterIndex1, clusterIndex2)
	
	local cluster1 = {clusters[clusterIndex1]}
	
	local cluster2 = {clusters[clusterIndex2]}
	
	local mergedCluster = AqwamMatrixLibrary:add(cluster1, cluster2)
	
	local newClusters = {}
	
	for i, cluster in ipairs(clusters) do
		
		if (i ~= clusterIndex1) and (i ~= clusterIndex2) then
			
			table.insert(newClusters, cluster)
			
		end
		
	end
	
	table.insert(newClusters, mergedCluster[1])
	
	return newClusters
	
end

local function areModelParametersMatricesEqualInSizeAndValues(ModelParameters, PreviousModelParameters)
	
	local areModelParametersEqual = false
	
	if (PreviousModelParameters == nil) then return areModelParametersEqual end
	
	if (#ModelParameters ~= #PreviousModelParameters) or (#ModelParameters[1] ~= #PreviousModelParameters[1]) then return areModelParametersEqual end
	
	areModelParametersEqual = AqwamMatrixLibrary:areMatricesEqual(ModelParameters, PreviousModelParameters)
	
	return areModelParametersEqual
	
end

function DivisiveHierarchicalModel.new(numberOfClusters, distanceFunction, linkageMethod, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	local NewDivisiveHierarchicalModel = BaseModel.new()
	
	setmetatable(NewDivisiveHierarchicalModel, DivisiveHierarchicalModel)
	
	NewDivisiveHierarchicalModel.highestCost = highestCost or defaultHighestCost
	
	NewDivisiveHierarchicalModel.lowestCost = lowestCost or defaultLowestCost
	
	NewDivisiveHierarchicalModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	NewDivisiveHierarchicalModel.linkageMethod = linkageMethod or defaultLinkageMethod
	
	NewDivisiveHierarchicalModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters
	
	NewDivisiveHierarchicalModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)
	
	return NewDivisiveHierarchicalModel
	
end

function DivisiveHierarchicalModel:setParameters(numberOfClusters, distanceFunction, linkageMethod, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	self.highestCost = highestCost or self.highestCost
	
	self.lowestCost = lowestCost or self.lowestCost
	
	self.distanceFunction = distanceFunction or self.distanceFunction
	
	self.linkageMethod = linkageMethod or self.linkageMethod
	
	self.numberOfClusters = numberOfClusters or self.numberOfClusters
	
	self.stopWhenModelParametersDoesNotChange =  self:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)
	
end

function DivisiveHierarchicalModel:train(featureMatrix)
	
	local clusterIndex1
	
	local clusterIndex2
	
	local maximumDistance
	
	local isOutsideCostBounds
	
	local numberOfIterations = 0
	
	local cost = 0
	
	local costArray = {}
	
	local PreviousModelParameters
	
	local areModelParametersEqual = false
	
	local clusters = AqwamMatrixLibrary:copy(featureMatrix)
	
	local maxNumberOfClusters
	
	if self.ModelParameters then

		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end

		clusters = AqwamMatrixLibrary:verticalConcatenate(self.ModelParameters, clusters)

	end
	
	if (self.numberOfClusters) then
		
		maxNumberOfClusters = math.min(self.numberOfClusters, #clusters)
		
	else
		
		maxNumberOfClusters = #clusters
		
	end
	
	repeat
		
		self:iterationWait()
		
		numberOfIterations += 1
		
		clusterIndex1, clusterIndex2, maximumDistance = findFarthestCluster(clusters, self.distanceFunction, self.linkageMethod)
		
		PreviousModelParameters = self.ModelParameters
		
		if (clusterIndex1) and (clusterIndex2) then
			
			clusters = splitCluster(clusters, clusterIndex1, clusterIndex2)
			
		end
		
		self.ModelParameters = clusters
		
		cost += maximumDistance
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
		areModelParametersEqual = areModelParametersMatricesEqualInSizeAndValues(self.ModelParameters, PreviousModelParameters)
		
		isOutsideCostBounds = (cost <= self.lowestCost) or (cost >= self.highestCost)
		
	until isOutsideCostBounds or (numberOfIterations == maxNumberOfClusters) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)
	
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
