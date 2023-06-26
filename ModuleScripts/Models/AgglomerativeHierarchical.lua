local BaseModel = require(script.Parent.BaseModel)

AgglomerativeHierarchicalModel = {}

AgglomerativeHierarchicalModel.__index = AgglomerativeHierarchicalModel

setmetatable(AgglomerativeHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = -math.huge

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "manhattan"

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


local function findClosestClusters(clusters, distanceFunction)
	
	local distance
	
	local minDistance = math.huge
	
	local clusterIndex1 = nil
		
	local clusterIndex2 = nil
	
	for i = 1, #clusters, 1 do
		
		for j = i + 1, #clusters, 1 do
			
			distance = calculateDistance({clusters[i]}, {clusters[j]}, distanceFunction)
			
			if (distance < minDistance) then
				
				minDistance = distance
				
				clusterIndex1 = i
				
				clusterIndex2 = j
				
			end
			
		end
		
	end
	
	return clusterIndex1, clusterIndex2, minDistance
	
end


local function mergeClusters(clusters, clusterIndex1, clusterIndex2)
	
	local newCluster = AqwamMatrixLibrary:add({clusters[clusterIndex1]}, {clusters[clusterIndex2]})
	
	newCluster = AqwamMatrixLibrary:divide(newCluster, 2)
	
	newCluster = newCluster[1]
	
	table.remove(clusters, clusterIndex2)
	
	table.remove(clusters, clusterIndex1)
	
	table.insert(clusters, newCluster)
	
	return clusters
	
end

local function areModelParametersMatricesEqualInSizeAndValues(ModelParameters, PreviousModelParameters)
	
	local areModelParametersEqual = false
	
	if (PreviousModelParameters == nil) then return areModelParametersEqual end

	if (#ModelParameters ~= #PreviousModelParameters) or (#ModelParameters[1] ~= #PreviousModelParameters[1]) then return areModelParametersEqual end

	areModelParametersEqual = AqwamMatrixLibrary:areMatricesEqual(ModelParameters, PreviousModelParameters)
	
	return areModelParametersEqual
	
end

function AgglomerativeHierarchicalModel.new(numberOfClusters, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	local NewAgglomerativeHierarchicalModel = BaseModel.new()
	
	setmetatable(NewAgglomerativeHierarchicalModel, AgglomerativeHierarchicalModel)

	NewAgglomerativeHierarchicalModel.highestCost = highestCost or defaultHighestCost
	
	NewAgglomerativeHierarchicalModel.lowestCost = lowestCost or defaultLowestCost

	NewAgglomerativeHierarchicalModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewAgglomerativeHierarchicalModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters
	
	NewAgglomerativeHierarchicalModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)
	
	return NewAgglomerativeHierarchicalModel
	
end

function AgglomerativeHierarchicalModel:setParameters(numberOfClusters, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)

	self.highestCost = highestCost or self.highestCost
	
	self.lowestCost = lowestCost or self.lowestCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.stopWhenModelParametersDoesNotChange =  self:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)

end

function AgglomerativeHierarchicalModel:train(featureMatrix)
	
	local clusterIndex1
	
	local clusterIndex2
	
	local minimumDistance
	
	local isOutsideCostBounds
	
	local numberOfIterations = 0
	
	local cost = 0
	
	local costArray = {}
	
	local PreviousModelParameters
	
	local areModelParametersEqual = false
	
	local clusters = AqwamMatrixLibrary:copy(featureMatrix)
	
	if self.ModelParameters then
		
		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end
		
		clusters = AqwamMatrixLibrary:verticalConcatenate(self.ModelParameters, clusters)
		
	end
	
	repeat
		
		self:iterationWait()
		
		numberOfIterations += 1
		
		clusterIndex1, clusterIndex2, minimumDistance = findClosestClusters(clusters, self.distanceFunction)
		
		PreviousModelParameters = self.ModelParameters
		
		clusters = mergeClusters(clusters, clusterIndex1, clusterIndex2)
		
		self.ModelParameters = clusters
		
		cost += minimumDistance

		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
		areModelParametersEqual = areModelParametersMatricesEqualInSizeAndValues(self.ModelParameters, PreviousModelParameters)
		
		isOutsideCostBounds = (cost <= self.lowestCost) or (cost >= self.highestCost)
		
	until isOutsideCostBounds or (#clusters == self.numberOfClusters) or (#clusters == 1) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)
	
	self.ModelParameters = clusters
	
	return costArray
	
end

function AgglomerativeHierarchicalModel:predict(featureMatrix)
	
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

return AgglomerativeHierarchicalModel
