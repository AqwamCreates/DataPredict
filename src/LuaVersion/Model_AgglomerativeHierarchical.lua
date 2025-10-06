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
	
	DO NOT REMOVE THIS TEXT WITHOUT PERMISSION!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

AgglomerativeHierarchicalModel = {}

AgglomerativeHierarchicalModel.__index = AgglomerativeHierarchicalModel

setmetatable(AgglomerativeHierarchicalModel, IterativeMethodBaseModel)

local defaultNumberOfCentroids = 1

local defaultDistanceFunction = "Euclidean"

local defaultLinkageFunction = "Minimum"

local defaultStopWhenModelParametersDoesNotChange = false

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

local function createCentroidDistanceMatrix(centroids, distanceFunction)

	local numberOfData = #centroids

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfData}, 0)

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfData, 1 do

			if (i ~= j) then -- Necessary, because for some reason math.pow(0, 2) gives 1 instead of zero. So skip this step when same centroids.

				distanceMatrix[i][j] = calculateDistance({centroids[i]}, {centroids[j]} , distanceFunction)

			end

		end

	end

	return distanceMatrix

end

local function createNewMergedDistanceMatrix(centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	local numberOfData = #centroidDistanceMatrix

	local newCentroidDistanceMatrix = {}

	for i = 1, numberOfData, 1 do

		if (i == centroidIndex1) or (i == centroidIndex2) then continue end

		local newCentroidDistanceVector = {}

		for j = 1, numberOfData, 1 do

			if (j == centroidIndex1) or (j == centroidIndex2) then continue end

			table.insert(newCentroidDistanceVector, centroidDistanceMatrix[i][j])

		end

		table.insert(newCentroidDistanceMatrix, newCentroidDistanceVector)

	end

	if (#newCentroidDistanceMatrix == 0) then return {{0}} end

	local newRow = {}

	for i = 1, #newCentroidDistanceMatrix[1], 1 do table.insert(newRow, 1, 0) end

	table.insert(newCentroidDistanceMatrix, 1, newRow)

	for i = 1, #newCentroidDistanceMatrix, 1 do table.insert(newCentroidDistanceMatrix[i], 1, 0) end

	return newCentroidDistanceMatrix

end

local function applyFunctionToFirstRowAndColumnOfDistanceMatrix(functionToApply, centroidDistanceMatrix, newCentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	local newColumnIndex = 2

	local newRowIndex = 2

	local numberOfcentroids = #centroidDistanceMatrix

	for column = 1, numberOfcentroids, 1 do

		if (column == centroidIndex1) or (column == centroidIndex2) then continue end

		local distance = functionToApply(centroidDistanceMatrix[centroidIndex1][column],  centroidDistanceMatrix[centroidIndex2][column])

		newCentroidDistanceMatrix[1][newColumnIndex] = distance

		newColumnIndex = newColumnIndex + 1

	end

	for row = 1, numberOfcentroids, 1 do

		if (row == centroidIndex1) or (row == centroidIndex2) then continue end

		local distance = functionToApply(centroidDistanceMatrix[row][centroidIndex1],  centroidDistanceMatrix[row][centroidIndex2])

		newCentroidDistanceMatrix[newRowIndex][1] = distance

		newRowIndex = newRowIndex + 1

	end

	return newCentroidDistanceMatrix

end

-----------------------------------------------------------------------------------------------------------------------

local function minimumLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	local newCentroidDistanceMatrix = createNewMergedDistanceMatrix(centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	newCentroidDistanceMatrix = applyFunctionToFirstRowAndColumnOfDistanceMatrix(math.min, centroidDistanceMatrix, newCentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	return newCentroidDistanceMatrix

end

local function maximumLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	local newCentroidDistanceMatrix = createNewMergedDistanceMatrix(centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	newCentroidDistanceMatrix = applyFunctionToFirstRowAndColumnOfDistanceMatrix(math.max, centroidDistanceMatrix, newCentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	return newCentroidDistanceMatrix

end

local function groupAverageLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	local weightedGroupAverage = function (x, y) return (x + y) / 2 end

	local newCentroidDistanceMatrix = createNewMergedDistanceMatrix(centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	newCentroidDistanceMatrix = applyFunctionToFirstRowAndColumnOfDistanceMatrix(weightedGroupAverage, centroidDistanceMatrix, newCentroidDistanceMatrix, centroidIndex1, centroidIndex2)

	return newCentroidDistanceMatrix

end

local function wardLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	local newCentroidDistanceMatrix = createCentroidDistanceMatrix(centroids, "Euclidean")

	for i = 2, #newCentroidDistanceMatrix,1 do

		newCentroidDistanceMatrix[1][i] = math.pow(newCentroidDistanceMatrix[1][i], 2)

		newCentroidDistanceMatrix[i][1] = math.pow(newCentroidDistanceMatrix[i][1], 2)

	end

	return newCentroidDistanceMatrix

end

-----------------------------------------------------------------------------------------------------------------------

local function findClosestCentroids(centroidDistanceMatrix)

	local distance

	local minimumCentroidDistance = math.huge

	local centroidIndex1 = nil

	local centroidIndex2 = nil

	for i = 1, #centroidDistanceMatrix, 1 do

		for j = 1, #centroidDistanceMatrix, 1 do

			distance = centroidDistanceMatrix[i][j]

			if (distance < minimumCentroidDistance) and (i~=j) then

				minimumCentroidDistance = distance

				centroidIndex1 = i

				centroidIndex2 = j

			end

		end

	end

	return centroidIndex1, centroidIndex2

end

local function updateDistanceMatrix(linkageFunction, centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	if (linkageFunction == "Minimum") then

		return minimumLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	elseif (linkageFunction == "Maximum") then

		return maximumLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	elseif (linkageFunction == "GroupAverage") then

		return groupAverageLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	elseif (linkageFunction == "Ward") then

		return wardLinkage(centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	else

		error("Invalid linkage function!")

	end

end

local function createNewCentroids(centroids, centroidIndex1Combine, centroidIndex2ToCombine)

	local centroid1 = {centroids[centroidIndex1Combine]}

	local centroid2 = {centroids[centroidIndex2ToCombine]}

	local combinedCentroid = AqwamTensorLibrary:add(centroid1, centroid2)

	local centroidToBeAdded = AqwamTensorLibrary:divide(combinedCentroid, 2)
	
	local isIndex1SmallerThanIndex2 = (centroidIndex1Combine < centroidIndex2ToCombine)
	
	local firstCentroidIndexToBeRemoved = (isIndex1SmallerThanIndex2 and centroidIndex2ToCombine) or centroidIndex1Combine
	
	local secondCentroidIndexToBeRemoved = (isIndex1SmallerThanIndex2 and centroidIndex1Combine) or centroidIndex2ToCombine
	
	table.remove(centroids, firstCentroidIndexToBeRemoved)
	
	table.remove(centroids, secondCentroidIndexToBeRemoved)

	table.insert(centroids, centroidToBeAdded[1])

	return centroids

end

local function calculateCost(centroids, featureMatrix, distanceFunction)

	local cost = 0

	for i = 1, #featureMatrix, 1 do

		local featureVector = {featureMatrix[i]}

		local minimumDistance = math.huge

		for j = 1, #centroids, 1 do

			local centroid = {centroids[j]}

			local distance = calculateDistance(featureVector, centroid, distanceFunction)

			minimumDistance = math.min(minimumDistance, distance)

		end

		cost = cost + minimumDistance

	end

	return cost

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

function AgglomerativeHierarchicalModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewAgglomerativeHierarchicalModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewAgglomerativeHierarchicalModel, AgglomerativeHierarchicalModel)
	
	NewAgglomerativeHierarchicalModel:setName("AgglomerativeHierarchical")

	NewAgglomerativeHierarchicalModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewAgglomerativeHierarchicalModel.linkageFunction = parameterDictionary.linkageFunction or defaultLinkageFunction

	NewAgglomerativeHierarchicalModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfCentroids

	return NewAgglomerativeHierarchicalModel

end

function AgglomerativeHierarchicalModel:train(featureMatrix)
	
	local numberOfClusters = self.numberOfClusters

	local distanceFunction = self.distanceFunction

	local linkageFunction = self.linkageFunction
	
	local ModelParameters = self.ModelParameters
	
	local centroidMatrix = AqwamTensorLibrary:copy(featureMatrix)
	
	local costArray = {}

	local numberOfIterations = 0

	local cost = 0

	local centroidDistanceMatrix

	local centroidIndex1

	local centroidIndex2

	if (ModelParameters) then

		if (#centroidMatrix[1] ~= #ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end

		centroidMatrix = AqwamTensorLibrary:concatenate(centroidMatrix, ModelParameters, 1)

	end

	centroidDistanceMatrix = createCentroidDistanceMatrix(centroidMatrix, distanceFunction)

	repeat
		
		numberOfIterations = numberOfIterations + 1

		self:iterationWait()
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(centroidMatrix, featureMatrix, distanceFunction)
			
		end)
		
		if cost then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)
			
		end

		centroidIndex1, centroidIndex2 = findClosestCentroids(centroidDistanceMatrix)
		
		centroidMatrix = createNewCentroids(centroidMatrix, centroidIndex1, centroidIndex2)

		centroidDistanceMatrix = updateDistanceMatrix(linkageFunction, centroidMatrix, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

	until (#centroidMatrix == numberOfClusters) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	self.ModelParameters = centroidMatrix

	return costArray

end

function AgglomerativeHierarchicalModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, self.ModelParameters, self.distanceFunction)

	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector

end

return AgglomerativeHierarchicalModel
