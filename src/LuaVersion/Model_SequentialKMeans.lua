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

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

SequentialKMeansModel = {}

SequentialKMeansModel.__index = SequentialKMeansModel

setmetatable(SequentialKMeansModel, IterativeMethodBaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local defaultMaximumNumberOfIterations = 1

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "Euclidean"

local defaultStopWhenModelParametersDoesNotChange = false

local defaultSetInitialCentroidsOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = false

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

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData1, numberOfData2})
	
	local calculateDistance = distanceFunctionList[distanceFunction]

	for matrix1Index = 1, numberOfData1, 1 do

		for matrix2Index = 1, numberOfData2, 1 do

			distanceMatrix[matrix1Index][matrix2Index] = calculateDistance({matrix1[matrix1Index]}, {matrix2[matrix2Index]})

		end

	end

	return distanceMatrix

end

local function chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, blacklistedDataIndexArray)
	
	local dataIndex

	local maxDistance = -math.huge

	for row = 1, #distanceMatrix, 1 do

		if table.find(blacklistedDataIndexArray, row) then continue end

		local totalDistance = 0

		for column = 1, #distanceMatrix[1], 1 do

			totalDistance = totalDistance + distanceMatrix[row][column]

		end

		if (totalDistance < maxDistance) then continue end

		maxDistance = totalDistance
		dataIndex = row

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

function SequentialKMeansModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewSequentialKMeansModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewSequentialKMeansModel, SequentialKMeansModel)
	
	NewSequentialKMeansModel:setName("SequentialKMeans")

	NewSequentialKMeansModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewSequentialKMeansModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters

	NewSequentialKMeansModel.setInitialCentroidsOnDataPoints =  NewSequentialKMeansModel:getValueOrDefaultValue(parameterDictionary.setInitialCentroidsOnDataPoints, defaultSetInitialCentroidsOnDataPoints)
	
	NewSequentialKMeansModel.setTheCentroidsDistanceFarthest = NewSequentialKMeansModel:getValueOrDefaultValue(parameterDictionary.setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	return NewSequentialKMeansModel
	
end

function SequentialKMeansModel:initializeCentroids(featureMatrix, numberOfClusters, distanceFunction, setInitialClustersOnDataPoints, setTheCentroidsDistanceFarthest)
	
	local ModelParameters
	
	if setInitialClustersOnDataPoints and setTheCentroidsDistanceFarthest then

		ModelParameters = chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	elseif setInitialClustersOnDataPoints and not setTheCentroidsDistanceFarthest then

		ModelParameters = chooseRandomCentroids(featureMatrix, numberOfClusters)

	else

		ModelParameters = self:initializeMatrixBasedOnMode({numberOfClusters, #featureMatrix[1]})

	end
	
	return ModelParameters
	
end

function SequentialKMeansModel:train(featureMatrix)
	
	local areModelParametersEqual
	
	local cost
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local distanceFunction = self.distanceFunction
	
	local ModelParameters = self.ModelParameters or {}
	
	local centroidMatrix = ModelParameters[1]
	
	local numberOfDataPointVector = ModelParameters[2]
	
	local cost
	
	if (centroidMatrix) then
		
		if (#featureMatrix[1] ~= #centroidMatrix[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		centroidMatrix = self:initializeCentroids(featureMatrix, numberOfClusters, self.distanceFunction, self.setInitialClustersOnDataPoints, self.setTheCentroidsDistanceFarthest)
		
	end
	
	repeat
		
		cost = math.huge
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		if (maximumNumberOfIterations > 1) then

			numberOfDataPointVector = AqwamTensorLibrary:createTensor({numberOfClusters, 1}, 0)

		end
		
		for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
			
			local featureVector = {unwrappedFeatureVector}
			
			local distanceVector = createDistanceMatrix(featureVector, centroidMatrix, distanceFunction)
			
			local minimumDistance = math.huge
			
			local clusterIndexWithMinimumDistance
			
			for clusterIndex = 1, numberOfClusters, 1 do
				
				local distance = distanceVector[1][clusterIndex]
				
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
			
			cost = self:calculateCostWhenRequired(numberOfIterations, function()

				return cost + minimumDistance

			end) 

			if (cost) then

				table.insert(costArray, cost)

				self:printNumberOfIterationsAndCost(numberOfIterations, cost)

			end

			if (self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)) then break end
			
		end

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.ModelParameters = {centroidMatrix, numberOfDataPointVector}
	
	return costArray
	
end

function SequentialKMeansModel:predict(featureMatrix, returnOriginalOutput)
	
	local centroidMatrix = self.ModelParameters[1]
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, self.distanceFunction)
	
	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return SequentialKMeansModel
