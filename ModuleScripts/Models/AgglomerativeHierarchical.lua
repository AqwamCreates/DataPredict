local BaseModel = require(script.Parent.BaseModel)

AgglomerativeHierarchicalModel = {}

AgglomerativeHierarchicalModel.__index = AgglomerativeHierarchicalModel

setmetatable(AgglomerativeHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = 0

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "manhattan"

local defaultStopWhenModelParametersDoesNotChange = false

local distanceFunctionList = {

	["manhattan"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:sum(part1)

		local distance = math.abs(part2)

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

	if (#ModelParameters ~= #PreviousModelParameters) and (#ModelParameters[1] ~= #PreviousModelParameters[1]) then return areModelParametersEqual end

	areModelParametersEqual = AqwamMatrixLibrary:areMatricesEqual(ModelParameters, PreviousModelParameters)
	
	return areModelParametersEqual
	
end

function AgglomerativeHierarchicalModel.new(maxNumberOfIterations, numberOfClusters, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	local NewAgglomerativeHierarchicalModel = BaseModel.new()
	
	setmetatable(NewAgglomerativeHierarchicalModel, AgglomerativeHierarchicalModel)
	
	NewAgglomerativeHierarchicalModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewAgglomerativeHierarchicalModel.highestCost = highestCost or defaultHighestCost
	
	NewAgglomerativeHierarchicalModel.lowestCost = lowestCost or defaultLowestCost

	NewAgglomerativeHierarchicalModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewAgglomerativeHierarchicalModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters
	
	NewAgglomerativeHierarchicalModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)
	
	return NewAgglomerativeHierarchicalModel
	
end

function AgglomerativeHierarchicalModel:setParameters(maxNumberOfIterations, numberOfClusters, distanceFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.highestCost = highestCost or self.highestCost
	
	self.lowestCost = lowestCost or self.lowestCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)

end


function AgglomerativeHierarchicalModel:train(featureMatrix)
	
	local clusterIndex1
	
	local clusterIndex2
	
	local minimumDistance
	
	local numberOfIterations = 0
	
	local cost = 0
	
	local costArray = {}
	
	local PreviousModelParameters
	
	local areModelParametersEqual = false
	
	local clusters = self.ModelParameters or AqwamMatrixLibrary:copy(featureMatrix)
	
	local isOutsideCostBounds
	
	repeat
		
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
		
	until (numberOfIterations == self.maxNumberOfIterations) or isOutsideCostBounds  or (#clusters == self.numberOfClusters) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)
	
	return costArray
	
end

function AgglomerativeHierarchicalModel:predict(featureMatrix)
	
	local distance
	
	local closestCluster
	
	local clusterVector
	
	local minimumDistance = math.huge
	
	for i, cluster in ipairs(self.ModelParameters) do
		
		clusterVector = {cluster}
		
		distance = calculateDistance(featureMatrix, self.distanceFunction)
		
		if (distance < minimumDistance) then
			
			minimumDistance = distance
			
			closestCluster = i
			
		end
		
	end

	return closestCluster, minimumDistance
	
end

return AgglomerativeHierarchicalModel
