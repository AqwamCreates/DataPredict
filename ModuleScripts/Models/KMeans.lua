local BaseModel = require(script.Parent.BaseModel)

KMeansModel = {}

KMeansModel.__index = KMeansModel

setmetatable(KMeansModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultTargetCost = 0

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "euclidean"

local defaultStopWhenModelParametersDoesNotChange = false

local defaultSetInitialClustersOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = false

local defaultLearningRate = 0.3

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

		local part3 = AqwamMatrixLibrary:applyFunction(math.sqrt, part2)

		local distance = AqwamMatrixLibrary:sum(part3)

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

local function calculateModelParametersMean(modelParameters, featureMatrix, distanceFunction, learningRate)
	
	local distanceMatrix = createDistanceMatrix(modelParameters, featureMatrix, distanceFunction)

	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix) 
	
	local sumOfAssignedCentroidVector = AqwamMatrixLibrary:verticalSum(clusterAssignmentMatrix) -- since row is the number of data in clusterAssignmentMatrix, then we vertical sum it
	
	local newModelParameters = AqwamMatrixLibrary:createMatrix(#modelParameters, #modelParameters[1])
	
	for cluster = 1, #modelParameters, 1 do
		
		sumOfAssignedCentroidVector[1][cluster] = math.max(1, sumOfAssignedCentroidVector[1][cluster])
		
		newModelParameters[cluster] = AqwamMatrixLibrary:divide({modelParameters[cluster]}, sumOfAssignedCentroidVector[1][cluster])[1]
		
	end
	
	local modelParametersWithLearningRate = AqwamMatrixLibrary:multiply(learningRate, AqwamMatrixLibrary:subtract(newModelParameters, modelParameters))
	
	modelParameters = AqwamMatrixLibrary:subtract(modelParameters, modelParametersWithLearningRate)
	
	return modelParameters
	
end

function KMeansModel.new(maxNumberOfIterations, learningRate, numberOfClusters, distanceFunction, targetCost, setInitialClustersOnDataPoints, setTheCentroidsDistanceFarthest, stopWhenModelParametersDoesNotChange)
	
	local NewKMeansModel = BaseModel.new()
	
	setmetatable(NewKMeansModel, KMeansModel)
	
	NewKMeansModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewKMeansModel.targetCost = targetCost or defaultTargetCost

	NewKMeansModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewKMeansModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters

	NewKMeansModel.learningRate = learningRate or defaultLearningRate
	
	NewKMeansModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)

	NewKMeansModel.setInitialClustersOnDataPoints =  BaseModel:getBooleanOrDefaultOption(setInitialClustersOnDataPoints, defaultSetInitialClustersOnDataPoints)
	
	NewKMeansModel.setTheCentroidsDistanceFarthest = BaseModel:getBooleanOrDefaultOption(setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	return NewKMeansModel
	
end

function KMeansModel:setParameters(maxNumberOfIterations, learningRate, numberOfClusters, distanceFunction, targetCost, setInitialClustersOnDataPoints, setTheCentroidsDistanceFarthest, stopWhenModelParametersDoesNotChange)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.targetCost = targetCost or self.targetCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.learningRate = learningRate or self.learningRate

	self.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)

	self.setInitialClustersOnDataPoints =  BaseModel:getBooleanOrDefaultOption(setInitialClustersOnDataPoints, self.setInitialClustersOnDataPoints)

	self.setTheCentroidsDistanceFarthest =  BaseModel:getBooleanOrDefaultOption(setTheCentroidsDistanceFarthest, self.setTheCentroidsDistanceFarthest)
	
end

local function initializeCentroids(featureMatrix, numberOfClusters, distanceFunction, setInitialClustersOnDataPoints, setTheCentroidsDistanceFarthest)
	
	local ModelParameters
	
	if setInitialClustersOnDataPoints and setTheCentroidsDistanceFarthest then

		ModelParameters = chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	elseif setInitialClustersOnDataPoints and not setTheCentroidsDistanceFarthest then

		ModelParameters = chooseRandomCentroids(featureMatrix, numberOfClusters)

	else

		ModelParameters = AqwamMatrixLibrary:createRandomMatrix(numberOfClusters, #featureMatrix[1])

	end
	
	return ModelParameters
	
end

function KMeansModel:train(featureMatrix)
	
	local PreviousModelParameters
	
	local areModelParametersEqual
	
	local cost
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	if (self.ModelParameters) then
		
		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		self.ModelParameters = initializeCentroids(featureMatrix, self.numberOfClusters, self.distanceFunction, self.setInitialClustersOnDataPoints, self.setTheCentroidsDistanceFarthest)
		
	end
	
	repeat

		numberOfIterations += 1
		
		PreviousModelParameters = self.ModelParameters

		self.ModelParameters = calculateModelParametersMean(self.ModelParameters, featureMatrix, self.distanceFunction, self.learningRate)

		areModelParametersEqual =  AqwamMatrixLibrary:areMatricesEqual(self.ModelParameters, PreviousModelParameters)
		
		cost = calculateCost(self.ModelParameters, featureMatrix, self.distanceFunction)
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	return costArray
	
end

function KMeansModel:predict(featureMatrix)
	
	local distanceFromClusterRowVector = createDistanceMatrix(self.ModelParameters, featureMatrix, self.distanceFunction)

	local clusterNumber, shortestDistance = assignToCluster(distanceFromClusterRowVector)
	
	return clusterNumber, shortestDistance
	
end

return KMeansModel
