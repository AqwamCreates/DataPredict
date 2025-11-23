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

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

local AgglomerativeHierarchicalModel = {}

AgglomerativeHierarchicalModel.__index = AgglomerativeHierarchicalModel

setmetatable(AgglomerativeHierarchicalModel, IterativeMethodBaseModel)

local defaultNumberOfCentroids = 1

local defaultDistanceFunction = "Euclidean"

local defaultLinkageFunction = "Minimum"

local defaultStopWhenModelParametersDoesNotChange = false

local function createCentroidDistanceMatrix(distanceFunction, centroidMatrix)

	local numberOfData = #centroidMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfData}, 0)

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfData, 1 do

			if (i ~= j) then -- Necessary, because for some reason math.pow(0, 2) gives 1 instead of zero. So skip this step when same centroids.

				distanceMatrix[i][j] = distanceFunction({centroidMatrix[i]}, {centroidMatrix[j]})
				
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

		if (column ~= centroidIndex1) and (column ~= centroidIndex2) then
			
			local distance = functionToApply(centroidDistanceMatrix[centroidIndex1][column],  centroidDistanceMatrix[centroidIndex2][column])

			newCentroidDistanceMatrix[1][newColumnIndex] = distance

			newColumnIndex = newColumnIndex + 1
			
		end
		
	end

	for row = 1, numberOfcentroids, 1 do

		if (row ~= centroidIndex1) and (row ~= centroidIndex2) then
			
			local distance = functionToApply(centroidDistanceMatrix[row][centroidIndex1],  centroidDistanceMatrix[row][centroidIndex2])

			newCentroidDistanceMatrix[newRowIndex][1] = distance

			newRowIndex = newRowIndex + 1
			
		end

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

	local newCentroidDistanceMatrix = createCentroidDistanceMatrix(centroids, distanceFunctionDictionary["Euclidean"])

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

local linkageFunctionList = {
	
	Minimum = minimumLinkage,
	
	Maximum = maximumLinkage,
	
	GroupAverage = groupAverageLinkage,
	
	Ward = wardLinkage,
	
}

local function mergeCentroids(centroids, centroidIndex1Combine, centroidIndex2ToCombine)

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

local function calculateCost(distanceFunction, featureMatrix, centroidMatrix)

	local cost = 0
	
	for _, unwrappedfeatureVector in ipairs(featureMatrix) do
		
		local featureVector = {unwrappedfeatureVector}

		local minimumDistance = math.huge
		
		for _, unwrappedCentroidVector in ipairs(centroidMatrix) do
			
			local centroidVector = {unwrappedCentroidVector}

			local distance = distanceFunction(featureVector, centroidVector)
			
			minimumDistance = math.min(minimumDistance, distance)
			
		end
		
		cost = cost + minimumDistance
		
	end

	return cost

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
	
	local linkageFunctionToApply = linkageFunctionList[linkageFunction]
	
	if (not linkageFunctionToApply) then error("Unknown linkage function.") end
	
	local distanceFunctionToApply = distanceFunctionDictionary[distanceFunction]
	
	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	local centroidMatrix = AqwamTensorLibrary:copy(featureMatrix)
	
	local costArray = {}

	local numberOfIterations = 0

	local cost = 0

	local centroidDistanceMatrix

	local centroidIndex1

	local centroidIndex2
	
	local numberOfCentroids

	if (ModelParameters) then

		if (#centroidMatrix[1] ~= #ModelParameters[1]) then error("The number of features are not the same as the model parameters.") end

		centroidMatrix = AqwamTensorLibrary:concatenate(centroidMatrix, ModelParameters, 1)

	end

	centroidDistanceMatrix = createCentroidDistanceMatrix(distanceFunctionToApply, centroidMatrix)

	repeat
		
		numberOfIterations = numberOfIterations + 1

		self:iterationWait()
		
		centroidIndex1, centroidIndex2 = findClosestCentroids(centroidDistanceMatrix)

		centroidMatrix = mergeCentroids(centroidMatrix, centroidIndex1, centroidIndex2)

		centroidDistanceMatrix = linkageFunctionToApply(centroidMatrix, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

		numberOfCentroids = #centroidMatrix
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(distanceFunctionToApply, featureMatrix, centroidMatrix)
			
		end)
		
		if (cost) then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)
			
		end

	until (numberOfCentroids == numberOfClusters) or (numberOfCentroids == 1) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = centroidMatrix

	return costArray

end

function AgglomerativeHierarchicalModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then

		local numberOfData = #featureMatrix

		if (returnOriginalOutput) then AqwamTensorLibrary:createTensor({numberOfData, self.numberOfClusters}, math.huge) end

		local dimensionSizeArray = {numberOfData, 1}

		return AqwamTensorLibrary:createTensor(dimensionSizeArray, nil), AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)

	end
	
	local distanceFunctionToApply = distanceFunctionDictionary[self.distanceFunction]
	
	local distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, ModelParameters)

	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector

end

return AgglomerativeHierarchicalModel
