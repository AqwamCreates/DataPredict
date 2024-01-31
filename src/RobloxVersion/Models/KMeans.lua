local BaseModel = require(script.Parent.BaseModel)

KMeansModel = {}

KMeansModel.__index = KMeansModel

setmetatable(KMeansModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "Euclidean"

local defaultStopWhenModelParametersDoesNotChange = false

local defaultSetInitialClustersOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = false

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local part3 = AqwamMatrixLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,

}


local function assignToCluster(distanceMatrix) -- Number of columns -> number of clusters
	
	local clusterNumberVector = AqwamMatrixLibrary:createMatrix(#distanceMatrix, 1)

	local clusterDistanceVector = AqwamMatrixLibrary:createMatrix(#distanceMatrix, 1) 

	for dataIndex, distanceVector in ipairs(distanceMatrix) do

		local closestClusterNumber

		local shortestDistance = math.huge

		for i, distance in ipairs(distanceVector) do

			if (distance < shortestDistance) then

				closestClusterNumber = i

				shortestDistance = distance

			end

		end

		clusterNumberVector[dataIndex][1] = closestClusterNumber

		clusterDistanceVector[dataIndex][1] = shortestDistance

	end

	return clusterNumberVector, clusterDistanceVector
	
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
	
	local calculateDistance = distanceFunctionList[distanceFunction]

	for datasetIndex = 1, #featureMatrix, 1 do

		for cluster = 1, #modelParameters, 1 do

			distanceMatrix[datasetIndex][cluster] = calculateDistance({featureMatrix[datasetIndex]}, {modelParameters[cluster]})

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
		
		local distanceVector = {distanceMatrix[dataIndex]}
		
		local _, vectorIndexArray = AqwamMatrixLibrary:findMaximumValueInMatrix(distanceVector)
		
		if (vectorIndexArray == nil) then continue end
		
		local clusterNumber = vectorIndexArray[2]
		
		clusterAssignmentMatrix[dataIndex][clusterNumber] = 1
		
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

local function calculateModelParametersMean(modelParameters, featureMatrix, distanceFunction)
	
	local distanceMatrix = createDistanceMatrix(modelParameters, featureMatrix, distanceFunction)

	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix) 
	
	local sumOfAssignedCentroidVector = AqwamMatrixLibrary:verticalSum(clusterAssignmentMatrix) -- since row is the number of data in clusterAssignmentMatrix, then we vertical sum it
	
	local newModelParameters = AqwamMatrixLibrary:createMatrix(#modelParameters, #modelParameters[1])
	
	for cluster = 1, #modelParameters, 1 do
		
		sumOfAssignedCentroidVector[1][cluster] = math.max(1, sumOfAssignedCentroidVector[1][cluster])
		
		newModelParameters[cluster] = AqwamMatrixLibrary:divide({modelParameters[cluster]}, sumOfAssignedCentroidVector[1][cluster])[1]
		
	end
	
	return newModelParameters
	
end

function KMeansModel.new(maxNumberOfIterations, numberOfClusters, distanceFunction, setInitialClustersOnDataPoints, setTheCentroidsDistanceFarthest)
	
	local NewKMeansModel = BaseModel.new()
	
	setmetatable(NewKMeansModel, KMeansModel)
	
	NewKMeansModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewKMeansModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewKMeansModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters

	NewKMeansModel.setInitialClustersOnDataPoints =  BaseModel:getBooleanOrDefaultOption(setInitialClustersOnDataPoints, defaultSetInitialClustersOnDataPoints)
	
	NewKMeansModel.setTheCentroidsDistanceFarthest = BaseModel:getBooleanOrDefaultOption(setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	return NewKMeansModel
	
end

function KMeansModel:setParameters(maxNumberOfIterations, numberOfClusters, distanceFunction, setInitialClustersOnDataPoints, setTheCentroidsDistanceFarthest)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.setInitialClustersOnDataPoints =  self:getBooleanOrDefaultOption(setInitialClustersOnDataPoints, self.setInitialClustersOnDataPoints)

	self.setTheCentroidsDistanceFarthest =  self:getBooleanOrDefaultOption(setTheCentroidsDistanceFarthest, self.setTheCentroidsDistanceFarthest)
	
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
		
		self:iterationWait()

		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(self.ModelParameters, featureMatrix, self.distanceFunction)
			
		end) 
		
		if cost then
			
			table.insert(costArray, cost)
			
			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
		end
		
		PreviousModelParameters = self.ModelParameters

		self.ModelParameters = calculateModelParametersMean(self.ModelParameters, featureMatrix, self.distanceFunction)

	until (numberOfIterations == self.maxNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	return costArray
	
end

function KMeansModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceMatrix = createDistanceMatrix(self.ModelParameters, featureMatrix, self.distanceFunction)
	
	if (returnOriginalOutput == true) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return KMeansModel
