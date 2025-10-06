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

local IterativeMethodBaseModel = require("Model_terativeMethodBaseModel")

KMedoidsModel = {}

KMedoidsModel.__index = KMedoidsModel

setmetatable(KMedoidsModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = math.huge

local defaultNumberOfClusters = 1

local defaultDistanceFunction = "Manhattan"

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

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

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

local function assignToCluster(distanceMatrix) -- Number of columns -> number of clusters
	
	local numberOfData = #distanceMatrix
	
	local clusterNumberVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0)

	local clusterDistanceVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 0) 

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

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClusters})

	for datasetIndex = 1, #featureMatrix, 1 do

		for cluster = 1, #modelParameters, 1 do

			distanceMatrix[datasetIndex][cluster] = distanceFunction({featureMatrix[datasetIndex]}, {modelParameters[cluster]})

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

	local clusterAssignmentMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClusters})

	local dataPointClusterNumber

	for dataIndex = 1, numberOfData, 1 do

		local distanceVector = {distanceMatrix[dataIndex]}

		local vectorIndexArray = AqwamTensorLibrary:findMinimumValueDimensionIndexArray(distanceVector)

		if (vectorIndexArray == nil) then continue end

		local clusterNumber = vectorIndexArray[2]

		clusterAssignmentMatrix[dataIndex][clusterNumber] = 1

	end

	return clusterAssignmentMatrix
	
end

local function calculateCost(modelParameters, featureMatrix, distanceFunction)
	
	local distanceMatrix = createDistanceMatrix(modelParameters, featureMatrix, distanceFunction)
	
	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix)
	
	local costMatrix = AqwamTensorLibrary:multiply(distanceMatrix, clusterAssignmentMatrix)
	
	local cost = AqwamTensorLibrary:sum(costMatrix)
	
	return cost
	
end

local function initializeCentroids(featureMatrix, numberOfClusters, distanceFunction, setTheCentroidsDistanceFarthest)

	if (setTheCentroidsDistanceFarthest) then

		return chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	else

		return chooseRandomCentroids(featureMatrix, numberOfClusters)

	end

end


function KMedoidsModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewKMedoidsModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewKMedoidsModel, KMedoidsModel)
	
	NewKMedoidsModel:setName("KMedoids")
	
	NewKMedoidsModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters

	NewKMedoidsModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewKMedoidsModel.setTheCentroidsDistanceFarthest = NewKMedoidsModel:getValueOrDefaultValue(parameterDictionary.setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	return NewKMedoidsModel
	
end

function KMedoidsModel:train(featureMatrix)
	
	local previousMedoid
	
	local previousCost

	local currentCost

	local costArray = {}

	local numberOfIterations = 0
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local distanceFunction = self.distanceFunction
	
	local setTheCentroidsDistanceFarthest = self.setTheCentroidsDistanceFarthest
	
	local ModelParameters = self.ModelParameters
	
	local distanceFunctionToApply = distanceFunctionList[distanceFunction]
	
	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	if (ModelParameters) then
		
		if (#featureMatrix[1] ~= #ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end
		
		currentCost = calculateCost(ModelParameters, featureMatrix, distanceFunctionToApply)
		
	else
		
		ModelParameters = initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply, setTheCentroidsDistanceFarthest)
		
		currentCost = math.huge
		
	end
	
	repeat
		
		self:iterationWait()
		
		for row = 1, #featureMatrix, 1 do
			
			self:dataWait()

			for medoid = 1, numberOfClusters, 1 do

				previousCost = currentCost
				
				previousMedoid = ModelParameters[medoid]

				ModelParameters[medoid] = featureMatrix[row]

				currentCost = calculateCost(ModelParameters, featureMatrix, distanceFunctionToApply)

				if (currentCost > previousCost) then

					ModelParameters[medoid] = previousMedoid

					currentCost = previousCost

				end
				
				numberOfIterations = numberOfIterations + 1

				table.insert(costArray, currentCost)

				self:printNumberOfIterationsAndCost(numberOfIterations, currentCost)

				if (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(currentCost) or self:checkIfConverged(currentCost) then break end

			end

			if (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(currentCost) or self:checkIfConverged(currentCost) then break end

		end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(currentCost) or self:checkIfConverged(currentCost)
	
	if (currentCost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.ModelParameters = ModelParameters
	
	return costArray
	
end

function KMedoidsModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceFunctionToApply = distanceFunctionList[self.distanceFunction]
	
	local ModelParameters = self.ModelParameters
	
	local distanceMatrix = createDistanceMatrix(ModelParameters, featureMatrix, distanceFunctionToApply)
	
	if (returnOriginalOutput == true) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return KMedoidsModel
