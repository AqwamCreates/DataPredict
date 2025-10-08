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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local IterativeMethodBaseModel = require(script.Parent.IterativeMethodBaseModel)

KMeansModel = {}

KMeansModel.__index = KMeansModel

setmetatable(KMeansModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultNumberOfClusters = 1

local defaultDistanceFunction = "Euclidean"

local defaultMode = "Hybrid"

local defaultSetInitialCentroidsOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = true

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		part1 = AqwamTensorLibrary:applyFunction(math.abs, part1)

		local distance = AqwamTensorLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		local part2 = AqwamTensorLibrary:power(part1, 2)

		local part3 = AqwamTensorLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,
	
	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamTensorLibrary:dotProduct(x1, AqwamTensorLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamTensorLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamTensorLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

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

local function checkIfTheDataPointClusterNumberBelongsToTheCluster(dataPointClusterNumber, cluster)
	
	if (dataPointClusterNumber == cluster) then
		
		return 1
		
	else
		
		return 0
		
	end
	
end

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

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
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, distanceFunction)
	
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

local function createClusterAssignmentMatrix(distanceMatrix) -- contains values of 0 and 1, where 0 is "does not belong to this cluster"
	
	local numberOfData = #distanceMatrix -- Number of rows
	
	local numberOfClusters = #distanceMatrix[1]
	
	local clusterAssignmentMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClusters})
	
	local dataPointClusterNumber
	
	for dataIndex = 1, numberOfData, 1 do
		
		local distanceVector = {distanceMatrix[dataIndex]}
		
		local vectorIndexArray, _ = AqwamTensorLibrary:findMinimumValueDimensionIndexArray(distanceVector)
		
		if (vectorIndexArray) then
			
			local clusterNumber = vectorIndexArray[2]

			clusterAssignmentMatrix[dataIndex][clusterNumber] = 1
			
		end
		
	end
	
	return clusterAssignmentMatrix
	
end

local function calculateCost(distanceMatrix, clusterAssignmentMatrix)
	
	local costMatrix = AqwamTensorLibrary:multiply(distanceMatrix, clusterAssignmentMatrix)
	
	local cost = AqwamTensorLibrary:sum(costMatrix)
	
	return cost
	
end

local function calculateMean(clusterAssignmentMatrix, centroidMatrix)
	
	local numberOfCentroids = #centroidMatrix
	
	local numberOfFeatures = #centroidMatrix[1]
	
	local sumOfAssignedCentroidVector = AqwamTensorLibrary:sum(clusterAssignmentMatrix, 1) -- since row is the number of data in clusterAssignmentMatrix, then we vertical sum it
	
	local newCentroidMatrix = AqwamTensorLibrary:createTensor({numberOfCentroids, numberOfFeatures})
	
	for cluster = 1, numberOfCentroids, 1 do
		
		sumOfAssignedCentroidVector[1][cluster] = math.max(1, sumOfAssignedCentroidVector[1][cluster])
		
		newCentroidMatrix[cluster] = AqwamTensorLibrary:divide({centroidMatrix[cluster]}, sumOfAssignedCentroidVector[1][cluster])[1]
		
	end
	
	return newCentroidMatrix
	
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
	
	elseif (setInitialCentroidsOnDataPoints) and (setTheCentroidsDistanceFarthest) then

		return chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	elseif (setInitialCentroidsOnDataPoints) and (not setTheCentroidsDistanceFarthest) then

		return chooseRandomCentroids(featureMatrix, numberOfClusters)

	else

		return self:initializeMatrixBasedOnMode({numberOfClusters, #featureMatrix[1]})

	end
	
end

local function batchKMeans(centroidMatrix, distanceMatrix)

	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix) -- data x clusters
	
	centroidMatrix = calculateMean(clusterAssignmentMatrix, centroidMatrix)
	
	return centroidMatrix, clusterAssignmentMatrix
	
end

local function sequentialKMeans(centroidMatrix, distanceMatrix, featureMatrix, numberOfDataPointVector)
	
	local numberOfData = #featureMatrix
	
	local numberOfClusters = #centroidMatrix
	
	local clusterAssignmentMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClusters}, 0) -- data x clusters
	
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
		
		clusterAssignmentMatrix[dataIndex][clusterIndexWithMinimumDistance] = 1

	end
	
	return centroidMatrix, clusterAssignmentMatrix
	
end

local kMeansFunctionList = {
	
	["Batch"] = batchKMeans,
	
	["Sequential"] = sequentialKMeans,
	
}

function KMeansModel:train(featureMatrix)
	
	local areModelParametersEqual
	
	local cost
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local distanceFunction = self.distanceFunction
	
	local mode = self.mode
	
	local ModelParameters = self.ModelParameters or {}

	local centroidMatrix = ModelParameters[1]

	local numberOfDataPointVector = ModelParameters[2]
	
	local clusterAssignmentMatrix
	
	local distanceMatrix

	if (mode == "Hybrid") then -- This must be always above the centroid initialization check. Otherwise it will think this is second training round despite it being the first one!
		
		mode = (centroidMatrix and numberOfDataPointVector and "Sequential") or "Batch"		

	end
	
	local kMeansFunction = kMeansFunctionList[mode]

	if (not kMeansFunction) then error("Unknown mode.") end
	
	local distanceFunctionToApply = distanceFunctionList[distanceFunction]

	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	if (mode == "Sequential") then
		
		numberOfDataPointVector = numberOfDataPointVector or AqwamTensorLibrary:createTensor({numberOfClusters, 1}, 0)
		
		maximumNumberOfIterations = 1 
		
	end
	
	if (centroidMatrix) then
		
		if (#featureMatrix[1] ~= #centroidMatrix[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		centroidMatrix = self:initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply)
		
	end
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, distanceFunctionToApply)

		centroidMatrix, clusterAssignmentMatrix = kMeansFunction(centroidMatrix, distanceMatrix, featureMatrix, numberOfDataPointVector)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return calculateCost(distanceMatrix, clusterAssignmentMatrix)

		end)
		
		if (cost) then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	numberOfDataPointVector = AqwamTensorLibrary:sum(clusterAssignmentMatrix, 1) -- 1 x clusters

	numberOfDataPointVector = AqwamTensorLibrary:transpose(numberOfDataPointVector) -- clusters x 1
	
	self.ModelParameters = {centroidMatrix, numberOfDataPointVector}
	
	return costArray
	
end

function KMeansModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceFunctionToApply = distanceFunctionList[self.distanceFunction]
	
	local centroidMatrix = self.ModelParameters[1]
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, distanceFunctionToApply)
	
	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return KMeansModel
