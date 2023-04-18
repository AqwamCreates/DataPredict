local BaseModel = require(script.Parent.BaseModel)

KMedoidsModel = {}

KMedoidsModel.__index = KMedoidsModel

setmetatable(KMedoidsModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = math.huge

local defaultTargetCost = 0

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "manhattan"

local defaultStopWhenModelParametersDoesNotChange = false

local defaultSetTheCentroidsDistanceFarthest = false

local distanceFunctionList = {

	["manhattan"] = function (y, h) return math.abs(y - h) end,

	["euclidean"] = function (y, h) return (y - h)^2 end,

}

local function calculateManhattanDistance(vector1, vector2)

	local distance = 0

	for row = 1, #vector1, 1 do

		distance += distanceFunctionList["manhattan"](vector1[row][1], vector2[row][1])

	end

	return distance

end

local function calculateEuclideanDistance(vector1, vector2)
	
	local squaredDistance = 0
	
	for row = 1, #vector1, 1 do

		squaredDistance += distanceFunctionList["euclidean"](vector1[row][1], vector2[row][1])

	end
	
	local distance = math.sqrt(squaredDistance)
	
	return distance
	
end

local function calculateDistance(vector1, vector2, distanceFunction)
	
	local distance
	
	vector1 = AqwamMatrixLibrary:transpose(vector1)
	vector2 = AqwamMatrixLibrary:transpose(vector2)

	if (distanceFunction == "euclidean") then
		
		distance = calculateEuclideanDistance(vector1, vector2)
		
	elseif (distanceFunction == "manhattan") then
		
		distance = calculateManhattanDistance(vector1, vector2)
		
	end
	
	return distance 
	
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

local function chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, blacklistedDataIndexArray)
	
	local distance

	local maxDistance = 0
	
	local dataIndex
	
	for row = 1, #distanceMatrix, 1 do
		
		if table.find(blacklistedDataIndexArray, row) then continue end

		for column = 1, #distanceMatrix[1], 1 do
			
			if table.find(blacklistedDataIndexArray, column) then continue end

			distance = distanceMatrix[row][column]
			
			if (distance > maxDistance) then
				
				distance = maxDistance
				dataIndex = row
				
			end

		end

	end
	
	return dataIndex
	
end

local function chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)
	
	local modelParameters = {}
	
	local dataIndexArray = {}
	
	local dataIndex
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, distanceFunction)
	
	repeat
		
		dataIndex = chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, dataIndexArray)
		
		table.insert(dataIndexArray, dataIndex)
		
	until (#dataIndexArray == numberOfClusters)
	
	for row = 1, numberOfClusters, 1 do
		
		dataIndex = dataIndexArray[row]
		
		table.insert(modelParameters, featureMatrix[dataIndex])
		
	end
	
	return modelParameters
	
end

local function chooseRandomCentroids(featureMatrix, numberOfClusters)

	local modelParameters = {}

	local numberOfRows = #featureMatrix

	local randomRow

	local selectedRows = {}

	local hasANewRandomRowChosen

	for cluster = 1, numberOfClusters, 1 do

		repeat

			randomRow = Random.new():NextInteger(1, numberOfRows)

			hasANewRandomRowChosen = not (table.find(selectedRows, randomRow))

			if hasANewRandomRowChosen then

				table.insert(selectedRows, randomRow)
				modelParameters[cluster] = featureMatrix[randomRow]

			end

		until hasANewRandomRowChosen

	end

	return modelParameters

end

local function createClusterAssignmentMatrix(distanceMatrix) -- contains values of 0 and 1, where 0 is "does not belong to this cluster"
	
	local numberOfData = #distanceMatrix -- Number of rows
	
	local numberOfClusters = #distanceMatrix[1]
	
	local clusterAssignmentMatrix = AqwamMatrixLibrary:createMatrix(#distanceMatrix, #distanceMatrix[1])
	
	local dataPointClusterNumber
	
	for dataIndex = 1, numberOfData, 1 do
		
		dataPointClusterNumber = assignToCluster({distanceMatrix[dataIndex]})
		
		for cluster = 1, numberOfClusters, 1 do

			clusterAssignmentMatrix[dataIndex][cluster] = checkIfTheDataPointClusterNumberBelongsToTheCluster(dataPointClusterNumber, cluster)

		end
		
	end
	
	return clusterAssignmentMatrix
	
end

local function calculateCost(modelParameters, featureMatrix, distanceFunction)
	
	local distanceMatrix = createDistanceMatrix(modelParameters, featureMatrix, distanceFunction)
	
	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix)
	
	local costMatrixSquareRoot = AqwamMatrixLibrary:multiply(distanceMatrix, clusterAssignmentMatrix)
	
	local costMatrix = AqwamMatrixLibrary:multiply(costMatrixSquareRoot, costMatrixSquareRoot)
	
	local cost = AqwamMatrixLibrary:sum(costMatrix)
	
	return cost
	
end

local function initializeCentroids(featureMatrix, numberOfClusters, distanceFunction, setTheCentroidsDistanceFarthest)

	local ModelParameters

	if setTheCentroidsDistanceFarthest then

		ModelParameters = chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	else

		ModelParameters = chooseRandomCentroids(featureMatrix, numberOfClusters)

	end

	return ModelParameters

end


function KMedoidsModel.new(maxNumberOfIterations, numberOfClusters, distanceFunction, targetCost, setTheCentroidsDistanceFarthest)
	
	local NewKMedoidsModel = BaseModel.new()
	
	setmetatable(NewKMedoidsModel, KMedoidsModel)
	
	NewKMedoidsModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewKMedoidsModel.targetCost = targetCost or defaultTargetCost

	NewKMedoidsModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewKMedoidsModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters
	
	NewKMedoidsModel.setTheCentroidsDistanceFarthest = BaseModel:getBooleanOrDefaultOption(setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	return NewKMedoidsModel
	
end

function KMedoidsModel:setParameters(maxNumberOfIterations, numberOfClusters, distanceFunction, targetCost, setTheCentroidsDistanceFarthest)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.targetCost = targetCost or self.targetCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.setTheCentroidsDistanceFarthest =  BaseModel:getBooleanOrDefaultOption(setTheCentroidsDistanceFarthest, self.setTheCentroidsDistanceFarthest)
	
end

function KMedoidsModel:train(featureMatrix)
	
	local distanceMatrix
	
	local PreviousModelParameters
	
	local areModelParametersEqual
	
	local previousCost
	
	local cost = math.huge
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local featureRowVector
	
	local medoidRowVector
	
	local areSameVectors
	
	if (self.ModelParameters) then
		
		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		self.ModelParameters = initializeCentroids(featureMatrix, self.numberOfClusters, self.distanceFunction, self.setTheCentroidsDistanceFarthest)
		
	end
	
	for iteration = 1, self.numberOfClusters, 1 do

		for row = 1, #featureMatrix, 1 do

			featureRowVector = {featureMatrix[row]}

			for medoid = 1, self.numberOfClusters, 1 do

				medoidRowVector = {self.ModelParameters[medoid]}

				areSameVectors = AqwamMatrixLibrary:areMatricesEqual(medoidRowVector, featureRowVector)

				if (areSameVectors) then continue end

				PreviousModelParameters = self.ModelParameters

				previousCost = cost

				self.ModelParameters[medoid] = featureRowVector[1]

				cost = calculateCost(self.ModelParameters, featureMatrix, self.distanceFunction)

				if (cost > previousCost) then

					self.ModelParameters = PreviousModelParameters

					cost = previousCost

				end

				table.insert(costArray, cost)

				numberOfIterations += 1

				BaseModel:printCostAndNumberOfIterations(cost, numberOfIterations, self.IsOutputPrinted)

				if (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost) then break end

			end

			if (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost) then break end

		end

		if (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost) then break end

	end
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	return costArray
	
end

function KMedoidsModel:predict(featureMatrix)
	
	local distanceFromClusterRowVector = createDistanceMatrix(self.ModelParameters, featureMatrix, self.distanceFunction)

	local clusterNumber, shortestDistance = assignToCluster(distanceFromClusterRowVector)
	
	return clusterNumber, shortestDistance
	
end

return KMedoidsModel
