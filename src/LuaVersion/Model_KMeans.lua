--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

local distanceFunctionDictionary = require("Core_DistanceFunctionDictionary")

KMeansModel = {}

KMeansModel.__index = KMeansModel

setmetatable(KMeansModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultNumberOfClusters = 1

local defaultDistanceFunction = "Euclidean"

local defaultMode = "Hybrid"

local defaultSetInitialCentroidsOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = true

local function assignToCluster(distanceMatrix) -- Number of columns -> number of clusters
	
	local numberOfDistances = #distanceMatrix
	
	local clusterNumberVector = AqwamTensorLibrary:createTensor({numberOfDistances, 1})

	local clusterDistanceVector = AqwamTensorLibrary:createTensor({numberOfDistances, 1}) 

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

local function createDistanceMatrix(distanceFunction, matrix1, matrix2)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData1, numberOfData2})

	for matrix1Index = 1, numberOfData1, 1 do

		for matrix2Index = 1, numberOfData2, 1 do

			distanceMatrix[matrix1Index][matrix2Index] = distanceFunction({matrix1[matrix1Index]}, {matrix2[matrix2Index]})

		end

	end

	return distanceMatrix

end

local function chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, blacklistedDataIndexArray)

	local dataIndex

	local maxDistance = -math.huge

	for row = 1, #distanceMatrix, 1 do

		if (not table.find(blacklistedDataIndexArray, row)) then

			local totalDistance = 0

			for column = 1, #distanceMatrix[1], 1 do totalDistance = totalDistance + distanceMatrix[row][column] end

			if (totalDistance > maxDistance) then

				maxDistance = totalDistance

				dataIndex = row

			end

		end

	end

	return dataIndex

end

local function chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)
	
	local centroidMatrix = {}
	
	local dataIndexArray = {}
	
	local dataIndex
	
	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix, featureMatrix)
	
	repeat
		
		dataIndex = chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, dataIndexArray)
		
		table.insert(dataIndexArray, dataIndex)
		
	until (#dataIndexArray == numberOfClusters)
	
	for row = 1, numberOfClusters, 1 do
		
		dataIndex = dataIndexArray[row]
		
		table.insert(centroidMatrix, featureMatrix[dataIndex])
		
	end
	
	return centroidMatrix
	
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

local function createClusterAssignmentArray(distanceMatrix)
	
	local numberOfClusters = #distanceMatrix[1]
	
	local clusterAssignmentArray = {}
	
	local dataPointClusterNumber
	
	local minimumDistance
	
	local index
	
	for dataIndex, unwrappedDistanceVector in ipairs(distanceMatrix) do
		
		minimumDistance = math.huge
		
		index = nil
		
		for clusterIndex, distance in ipairs(unwrappedDistanceVector) do
			
			if (distance < minimumDistance) then
				
				minimumDistance = distance
				
				index = clusterIndex
			end
			
		end
		
		clusterAssignmentArray[dataIndex] = index or math.random(1, numberOfClusters)
		
	end
	
	return clusterAssignmentArray
	
end

local function calculateCost(distanceMatrix, clusterAssignmentArray)
	
	local cost = 0
	
	local clusterIndex
	
	for dataIndex, unwrappedDistanceVector in ipairs(distanceMatrix) do
		
		clusterIndex = clusterAssignmentArray[dataIndex]
		
		cost = cost + unwrappedDistanceVector[clusterIndex]
		
	end
	
	cost = cost / #distanceMatrix
	
	return cost
	
end

local function calculateMean(featureMatrix, numberOfClusters, clusterAssignmentArray)
	
	local numberOfFeatures = #featureMatrix[1]
	
	local centroidMatrix = AqwamTensorLibrary:createTensor({numberOfClusters, numberOfFeatures}, 0)
	
	local clusterCountArray = table.create(numberOfClusters, 0)
	
	local clusterIndex
	
	local clusterCount
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		clusterIndex = clusterAssignmentArray[dataIndex]
		
		centroidMatrix[clusterIndex] = AqwamTensorLibrary:add({centroidMatrix[clusterIndex]}, {unwrappedFeatureVector})[1]
		
		clusterCountArray[clusterIndex] = clusterCountArray[clusterIndex] + 1
		
	end
	
	for clusterIndex, unwrappedCentroidVector in ipairs(centroidMatrix) do
		
		clusterCount = clusterCountArray[clusterIndex]
		
		if (clusterCount ~= 0) then
			
			centroidMatrix[clusterIndex] = AqwamTensorLibrary:divide({unwrappedCentroidVector}, clusterCount)[1]
			
		end
		
	end
	
	return centroidMatrix
	
end

function KMeansModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewKMeansModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewKMeansModel, KMeansModel)
	
	NewKMeansModel:setName("KMeans")

	NewKMeansModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters
	
	NewKMeansModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewKMeansModel.mode = parameterDictionary.mode or defaultMode

	NewKMeansModel.setInitialCentroidsOnDataPoints =  NewKMeansModel:getValueOrDefaultValue(parameterDictionary.setInitialCentroidsOnDataPoints, defaultSetInitialCentroidsOnDataPoints)
	
	NewKMeansModel.setTheCentroidsDistanceFarthest = NewKMeansModel:getValueOrDefaultValue(parameterDictionary.setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	return NewKMeansModel
	
end

function KMeansModel:initializeCentroids(featureMatrix, numberOfClusters, distanceFunction)
	
	local setInitialCentroidsOnDataPoints = self.setInitialCentroidsOnDataPoints
	
	local setTheCentroidsDistanceFarthest = self.setTheCentroidsDistanceFarthest
	
	if (setInitialCentroidsOnDataPoints) and (numberOfClusters == 1) then
		
		return AqwamTensorLibrary:mean(featureMatrix, 1)
	
	elseif (setInitialCentroidsOnDataPoints) and (setTheCentroidsDistanceFarthest) and (#featureMatrix >= numberOfClusters) then

		return chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	elseif (setInitialCentroidsOnDataPoints) and (not setTheCentroidsDistanceFarthest) then

		return chooseRandomCentroids(featureMatrix, numberOfClusters)

	else

		return self:initializeMatrixBasedOnMode({numberOfClusters, #featureMatrix[1]})

	end
	
end

local function batchKMeans(featureMatrix, centroidMatrix, distanceMatrix)

	local clusterAssignmentArray = createClusterAssignmentArray(distanceMatrix) -- data x clusters
	
	centroidMatrix = calculateMean(featureMatrix, #centroidMatrix, clusterAssignmentArray)
	
	return centroidMatrix, clusterAssignmentArray
	
end

local function sequentialKMeans(featureMatrix, centroidMatrix, distanceMatrix, numberOfDataPointVector)
	
	local numberOfData = #featureMatrix
	
	local numberOfClusters = #centroidMatrix
	
	local clusterAssignmentArray = {}
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do

		local featureVector = {unwrappedFeatureVector}

		local minimumDistance = math.huge

		local clusterIndexWithMinimumDistance

		for clusterIndex = 1, numberOfClusters, 1 do

			local distance = distanceMatrix[dataIndex][clusterIndex]

			if (distance < minimumDistance) then

				minimumDistance = distance

				clusterIndexWithMinimumDistance = clusterIndex

			end

		end

		local numberOfDataPoints = numberOfDataPointVector[clusterIndexWithMinimumDistance][1] + 1

		local centroidVector = {centroidMatrix[clusterIndexWithMinimumDistance]}

		local centroidChangeVectorPart1 = AqwamTensorLibrary:subtract(featureVector, centroidVector)

		local centroidChangeVector = AqwamTensorLibrary:multiply((1 / numberOfDataPoints), centroidChangeVectorPart1)

		local newCentroidVector = AqwamTensorLibrary:add(centroidVector, centroidChangeVector)

		numberOfDataPointVector[clusterIndexWithMinimumDistance][1] = numberOfDataPoints

		centroidMatrix[clusterIndexWithMinimumDistance] = newCentroidVector[1]
		
		clusterAssignmentArray[dataIndex] = clusterIndexWithMinimumDistance

	end
	
	return centroidMatrix, clusterAssignmentArray
	
end

local function createNumberOfDataPointVector(numberOfClusters, clusterAssignmentArray)
	
	local numberOfDataPointArray = table.create(numberOfClusters, 0) 

	for dataIndex, clusterAssignmentIndex in ipairs(clusterAssignmentArray) do

		numberOfDataPointArray[clusterAssignmentIndex] = numberOfDataPointArray[clusterAssignmentIndex] + 1

	end

	local numberOfDataPointVector = {}
	
	for clusterIndex, numberOfDataPoint in ipairs(numberOfDataPointArray) do
		
		numberOfDataPointVector[clusterIndex] = {numberOfDataPoint}
		
	end
	
	return numberOfDataPointVector
	
end

local kMeansFunctionList = {
	
	["Batch"] = batchKMeans,
	
	["Sequential"] = sequentialKMeans,
	
}

function KMeansModel:train(featureMatrix)
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local distanceFunction = self.distanceFunction
	
	local mode = self.mode
	
	local ModelParameters = self.ModelParameters or {}

	local centroidMatrix = ModelParameters[1]

	local numberOfDataPointVector = ModelParameters[2]

	if (mode == "Hybrid") then -- This must be always above the centroid initialization check. Otherwise it will think this is second training round despite it being the first one!
		
		mode = (centroidMatrix and numberOfDataPointVector and "Sequential") or "Batch"		

	end
	
	local kMeansFunction = kMeansFunctionList[mode]

	if (not kMeansFunction) then error("Unknown mode.") end
	
	local distanceFunctionToApply = distanceFunctionDictionary[distanceFunction]

	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	if (mode == "Sequential") then
		
		numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClusters, 1}, 0)
		
		maximumNumberOfIterations = 1 
		
	end
	
	if (centroidMatrix) then
		
		if (#featureMatrix[1] ~= #centroidMatrix[1]) then error("The number of features are not the same as the model parameters.") end
		
	else
		
		centroidMatrix = self:initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply)
		
	end

	local numberOfIterations = 0
	
	local costArray = {}
	
	local clusterAssignmentArray

	local distanceMatrix
	
	local cost
	
	local numberOfDataPointArray
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, centroidMatrix)

		centroidMatrix, clusterAssignmentArray = kMeansFunction(featureMatrix, centroidMatrix, distanceMatrix, numberOfDataPointVector)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return calculateCost(distanceMatrix, clusterAssignmentArray)

		end)
		
		if (cost) then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	numberOfDataPointVector = createNumberOfDataPointVector(numberOfClusters, clusterAssignmentArray)
	
	self.ModelParameters = {centroidMatrix, numberOfDataPointVector}
	
	return costArray
	
end

function KMeansModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceFunctionToApply = distanceFunctionDictionary[self.distanceFunction]
	
	local ModelParameters = self.ModelParameters
	
	local centroidMatrix
	
	if (not ModelParameters) then
		
		local numberOfClusters = self.numberOfClusters

		centroidMatrix = self:initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply)
		
		local numberOfDataPointVector = AqwamTensorLibrary:createTensor({numberOfClusters, 1})
		
		self.ModelParameters = {centroidMatrix, numberOfDataPointVector}
		
	else
		
		centroidMatrix = ModelParameters[1]

	end
	
	local distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, centroidMatrix)
	
	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return KMeansModel
