local BaseModel = require(script.Parent.BaseModel)

DensityBasedSpatialClusteringOfApplicationsWithNoiseModel = {}

DensityBasedSpatialClusteringOfApplicationsWithNoiseModel.__index = DensityBasedSpatialClusteringOfApplicationsWithNoiseModel

setmetatable(DensityBasedSpatialClusteringOfApplicationsWithNoiseModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultTargetCost = 0

local defaultMinimumNumberOfPoints = 2

local defaultDistanceFunction = "manhattan"

local defaultEpsilon = 10

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


local function getNeighbors(currentCorePointNumber, featureMatrix, epsilon, distanceFunction)
	
	local distance
	
	local neighbors = {}
	
	for i = 1, #featureMatrix, 1 do
		
		if (i ~= currentCorePointNumber) then
			
			distance = calculateDistance({featureMatrix[currentCorePointNumber]}, {featureMatrix[i]}, distanceFunction)
			
			if (distance <= epsilon) then
				
				neighbors[#neighbors + 1] = i
				
			end
			
		end
		
	end
	
	return neighbors
	
end

local function mergeTables(table1, table2)
	
	for i=1, #table2, 1 do
		
		table1[#table1+1] = table2[i]
		
	end
	
	return table1
	
end

local function expandCluster(currentCorePointNumber, neighbors, neighbouringCorePointNumber, clusters, visited, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	clusters[neighbouringCorePointNumber] = clusters[neighbouringCorePointNumber] or {}
	
	clusters[neighbouringCorePointNumber][#clusters[neighbouringCorePointNumber] + 1] = currentCorePointNumber
	
	for i = 1, #neighbors do
		
		local neighbouringPointNumber = neighbors[i]
		
		if not visited[neighbouringPointNumber] then
			
			visited[neighbouringPointNumber] = true
			
			local qNeighbors = getNeighbors(neighbouringPointNumber, featureMatrix, epsilon, distanceFunction)
			
			if (#qNeighbors >= minimumNumberOfPoints) then
				
				neighbors = mergeTables(neighbors, qNeighbors)
				
			end
			
		end
		
		local isInCluster = false
		
		for j = 1, #clusters do
			
			if (clusters[j][neighbouringPointNumber]) then
				
				isInCluster = true
				
				break
				
			end
			
		end
		
		if not isInCluster then
			
			clusters[neighbouringCorePointNumber][#clusters[neighbouringCorePointNumber] + 1] = neighbouringPointNumber
			
		end
		
	end
	
end

local function calculateCost(featureMatrix, clusters, distanceFunction)
	
	local cost = 0
	
	for cluster_id, clusterPoints in pairs(clusters) do
			
		for i = 1, #clusterPoints, 1 do
				
			for j = i + 1, #clusterPoints, 1 do
					
				cost = cost + calculateDistance({featureMatrix[clusterPoints[i]]}, {featureMatrix[clusterPoints[j]]}, distanceFunction)
					
			end
				
		end
		
	end
	
	return cost
	
end


function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel.new(epsilon, minimumNumberOfPoints, distanceFunction, targetCost)
	
	local NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel = BaseModel.new()
	
	setmetatable(NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel, DensityBasedSpatialClusteringOfApplicationsWithNoiseModel)
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.minimumNumberOfPoints = minimumNumberOfPoints or defaultMinimumNumberOfPoints
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.epsilon = epsilon or defaultEpsilon

	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.targetCost = targetCost or defaultTargetCost

	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.appendPreviousFeatureMatrix = false
	
	return NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:setParameters(epsilon, minimumNumberOfPoints, distanceFunction, targetCost)
	
	self.minimumNumberOfPoints = minimumNumberOfPoints or defaultMinimumNumberOfPoints

	self.epsilon = epsilon or defaultEpsilon

	self.targetCost = targetCost or defaultTargetCost

	self.distanceFunction = distanceFunction or defaultDistanceFunction
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:canAppendPreviousFeatureMatrix(option)

	self.appendPreviousFeatureMatrix = option

end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:train(featureMatrix)
	
	if (self.ModelParameters) and (self.appendPreviousFeatureMatrix) then

		local storedFeatureMatrix = self.ModelParameters[1]

		if (#storedFeatureMatrix[1] ~= #featureMatrix[1]) then error("The previous and current feature matrices do not have the same number of features.") end 

		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, storedFeatureMatrix)

	end
	
	local cost
	
	local neighbouringCorePointNumber 
	
	local neighbors 

	local costArray = {}

	local visited = {}
	
	local clusters = {}
	
	local visited = {}
	
	local noise = {}
	
	local numberOfData = #featureMatrix

	for currentCorePointNumber = 1, numberOfData, 1 do
		
		self:iterationWait()
		
		if not visited[currentCorePointNumber] then
			
			visited[currentCorePointNumber] = true
			
			neighbors = getNeighbors(currentCorePointNumber, featureMatrix, self.epsilon, self.distanceFunction)
			
			if (#neighbors < self.minimumNumberOfPoints) then
				
				noise[#noise + 1] = currentCorePointNumber
				
			else
				
				neighbouringCorePointNumber  = #clusters + 1
				
				expandCluster(currentCorePointNumber, neighbors, neighbouringCorePointNumber , clusters, visited, featureMatrix, self.epsilon, self.minimumNumberOfPoints, self.distanceFunction)
				
			end
			
		end
		
		cost = calculateCost(featureMatrix, clusters, self.distanceFunction)

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, currentCorePointNumber)
		
		if (cost == self.targetCost) then break end
		
	end
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.ModelParameters = {featureMatrix, clusters}
	
	return costArray
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:predict(featureMatrix)
	
	local shortestDistanceVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)

	local closestClusterVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	local storedFeatureVector, cluster = table.unpack(self.ModelParameters)
	
	for i = 1, #featureMatrix, 1 do
		
		local closestCluster

		local shortestDistance = math.huge
		
		local featureVector = {featureMatrix[i]}

		for clusterNumber, clusterPoints in ipairs(cluster) do

			local distance = 0

			for j = 1, #clusterPoints, 1 do

				local pointNumber = clusterPoints[j]

				local point = {storedFeatureVector[pointNumber]}

				distance += calculateDistance(featureVector, point, self.distanceFunction)

			end

			distance = distance / #clusterPoints

			if (distance < shortestDistance) then
				
				closestCluster = clusterNumber

				shortestDistance = distance

			end

		end
		
		closestClusterVector[i][1] = closestCluster
		
		shortestDistanceVector[i][1] = shortestDistance
		
	end
	
	return closestClusterVector, shortestDistanceVector
	
end

return DensityBasedSpatialClusteringOfApplicationsWithNoiseModel
