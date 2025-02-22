local BaseModel = require(script.Parent.BaseModel)

DensityBasedSpatialClusteringOfApplicationsWithNoiseModel = {}

DensityBasedSpatialClusteringOfApplicationsWithNoiseModel.__index = DensityBasedSpatialClusteringOfApplicationsWithNoiseModel

setmetatable(DensityBasedSpatialClusteringOfApplicationsWithNoiseModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMinimumNumberOfPoints = 2

local defaultDistanceFunction = "Manhattan"

local defaultEpsilon = 10

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)
		
		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		local part2 = AqwamMatrixLibrary:sum(part1)
		
		local distance = math.abs(part2)
		
		return distance 
		
	end,

	["Euclidean"] = function (x1, x2)
		
		local part1 = AqwamMatrixLibrary:subtract(x1, x2)
		
		local part2 = AqwamMatrixLibrary:power(part1, 2)
		
		local part3 = AqwamMatrixLibrary:sum(part2)
		
		local distance = math.sqrt(part3)
		
		return distance 
		
	end,
	
	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamMatrixLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamMatrixLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamMatrixLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamMatrixLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2, 2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

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

local function expandCluster(currentCorePointNumber, neighbors, neighbouringCorePointNumber, clusters, hasVisitedCorePointNumberArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	clusters[neighbouringCorePointNumber] = clusters[neighbouringCorePointNumber] or {}
	
	clusters[neighbouringCorePointNumber][#clusters[neighbouringCorePointNumber] + 1] = currentCorePointNumber
	
	for i = 1, #neighbors do
		
		local neighbouringPointNumber = neighbors[i]
		
		if (not hasVisitedCorePointNumberArray[neighbouringPointNumber]) then
			
			hasVisitedCorePointNumberArray[neighbouringPointNumber] = true
			
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
		
		if (not isInCluster) then
			
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


function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel.new(epsilon, minimumNumberOfPoints, distanceFunction)
	
	local NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel = BaseModel.new()
	
	setmetatable(NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel, DensityBasedSpatialClusteringOfApplicationsWithNoiseModel)
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.minimumNumberOfPoints = minimumNumberOfPoints or defaultMinimumNumberOfPoints
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.epsilon = epsilon or defaultEpsilon

	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	return NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:setParameters(epsilon, minimumNumberOfPoints, distanceFunction)
	
	self.minimumNumberOfPoints = minimumNumberOfPoints or defaultMinimumNumberOfPoints

	self.epsilon = epsilon or defaultEpsilon

	self.distanceFunction = distanceFunction or defaultDistanceFunction
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:train(featureMatrix)
	
	if (self.ModelParameters) then

		local storedFeatureMatrix = self.ModelParameters[1]

		if (#storedFeatureMatrix[1] ~= #featureMatrix[1]) then error("The previous and current feature matrices do not have the same number of features.") end 

		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, storedFeatureMatrix)

	end
	
	local cost
	
	local neighbouringCorePointNumber 
	
	local neighbors 

	local costArray = {}
	
	local clusters = {}
	
	local noiseCorePointNumberArray = {}
	
	local hasVisitedCorePointNumberArray = {}
	
	local numberOfData = #featureMatrix
	
	local minimumNumberOfPoints = self.minimumNumberOfPoints

	local epsilon = self.epsilon

	local distanceFunction = self.distanceFunction

	for currentCorePointNumber = 1, numberOfData, 1 do
		
		self:iterationWait()
		
		if (not hasVisitedCorePointNumberArray[currentCorePointNumber]) then
			
			hasVisitedCorePointNumberArray[currentCorePointNumber] = true
			
			neighbors = getNeighbors(currentCorePointNumber, featureMatrix, epsilon, distanceFunction)
			
			if (#neighbors < self.minimumNumberOfPoints) then
				
				table.insert(noiseCorePointNumberArray, currentCorePointNumber)
				
			else
				
				neighbouringCorePointNumber  = #clusters + 1
				
				expandCluster(currentCorePointNumber, neighbors, neighbouringCorePointNumber, clusters, hasVisitedCorePointNumberArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
				
			end
			
		end
		
		cost = self:calculateCostWhenRequired(currentCorePointNumber, function()
			
			return calculateCost(featureMatrix, clusters, distanceFunction)
			
		end)
		
		if cost then
			
			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, currentCorePointNumber)

			if self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) then break end
			
		end
		
	end
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.ModelParameters = {featureMatrix, clusters}
	
	return costArray
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:predict(featureMatrix)
	
	local shortestDistanceVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)

	local closestClusterVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	local storedFeatureVector, cluster = table.unpack(self.ModelParameters)
	
	local distanceFunction = self.distanceFunction
	
	for i = 1, #featureMatrix, 1 do
		
		local closestCluster

		local shortestDistance = math.huge
		
		local featureVector = {featureMatrix[i]}

		for clusterNumber, clusterPoints in ipairs(cluster) do

			local distance = 0

			for j = 1, #clusterPoints, 1 do

				local pointNumber = clusterPoints[j]

				local pointVector = {storedFeatureVector[pointNumber]}

				distance += calculateDistance(featureVector, pointVector, distanceFunction)

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
