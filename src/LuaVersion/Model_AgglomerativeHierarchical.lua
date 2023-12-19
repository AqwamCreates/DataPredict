--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

AgglomerativeHierarchicalModel = {}

AgglomerativeHierarchicalModel.__index = AgglomerativeHierarchicalModel

setmetatable(AgglomerativeHierarchicalModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

local defaultHighestCost = math.huge

local defaultLowestCost = -math.huge

local defaultNumberOfcentroids = 1

local defaultDistanceFunction = "Euclidean"

local defaultLinkageFunction = "Minimum"

local defaultStopWhenModelParametersDoesNotChange = false

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


local function calculateDistance(vector1, vector2, distanceFunction)

	return distanceFunctionList[distanceFunction](vector1, vector2) 

end

local function createCentroidDistanceMatrix(centroids, distanceFunction)

	local numberOfData = #centroids

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)

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

		newColumnIndex += 1

	end

	for row = 1, numberOfcentroids, 1 do

		if (row == centroidIndex1) or (row == centroidIndex2) then continue end

		local distance = functionToApply(centroidDistanceMatrix[row][centroidIndex1],  centroidDistanceMatrix[row][centroidIndex2])

		newCentroidDistanceMatrix[newRowIndex][1] = distance

		newRowIndex += 1

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

	local newCentroidDistanceMatrix = createCentroidDistanceMatrix(centroids, "euclidean")

	for i = 2, #newCentroidDistanceMatrix,1 do

		newCentroidDistanceMatrix[1][i] = math.pow(newCentroidDistanceMatrix[1][i], 2)

		newCentroidDistanceMatrix[i][1] = math.pow(newCentroidDistanceMatrix[i][1], 2)

	end

	return newCentroidDistanceMatrix

end

-----------------------------------------------------------------------------------------------------------------------

local function findClosestcentroids(centroidDistanceMatrix)

	local distance

	local minimumCentroidDistance = -math.huge

	local centroidIndex1 = nil

	local centroidIndex2 = nil

	for i = 1, #centroidDistanceMatrix, 1 do

		for j = 1, #centroidDistanceMatrix, 1 do

			distance = centroidDistanceMatrix[i][j]

			if (distance > minimumCentroidDistance) and (i~=j) then

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

local function createNewcentroids(centroids, centroidIndex1Combine, centroidIndex2ToCombine)

	local newcentroids = {}

	local centroid1 = {centroids[centroidIndex1Combine]}

	local centroid2 = {centroids[centroidIndex2ToCombine]}

	local combinedCentroid = AqwamMatrixLibrary:add(centroid1, centroid2)

	local centroidToBeAdded = AqwamMatrixLibrary:divide(combinedCentroid, 2)

	table.insert(newcentroids, centroidToBeAdded[1])

	for i = 1, #centroids, 1 do

		if (i ~= centroidIndex1Combine) and (i ~= centroidIndex2ToCombine) then table.insert(newcentroids, centroids[i]) end

	end

	return newcentroids

end

local function areModelParametersMatricesEqualInSizeAndValues(ModelParameters, PreviousModelParameters)

	local areModelParametersEqual = false

	if (PreviousModelParameters == nil) then return areModelParametersEqual end

	if (#ModelParameters ~= #PreviousModelParameters) or (#ModelParameters[1] ~= #PreviousModelParameters[1]) then return areModelParametersEqual end

	areModelParametersEqual = AqwamMatrixLibrary:areMatricesEqual(ModelParameters, PreviousModelParameters)

	return areModelParametersEqual

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

		cost += minimumDistance

	end

	return cost

end

function AgglomerativeHierarchicalModel.new(numberOfCentroids, distanceFunction, linkageFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)

	local NewAgglomerativeHierarchicalModel = BaseModel.new()

	setmetatable(NewAgglomerativeHierarchicalModel, AgglomerativeHierarchicalModel)

	NewAgglomerativeHierarchicalModel.highestCost = highestCost or defaultHighestCost

	NewAgglomerativeHierarchicalModel.lowestCost = lowestCost or defaultLowestCost

	NewAgglomerativeHierarchicalModel.distanceFunction = distanceFunction or defaultDistanceFunction

	NewAgglomerativeHierarchicalModel.linkageFunction = linkageFunction or defaultLinkageFunction

	NewAgglomerativeHierarchicalModel.numberOfCentroids = numberOfCentroids or defaultNumberOfcentroids

	NewAgglomerativeHierarchicalModel.stopWhenModelParametersDoesNotChange =  BaseModel:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, defaultStopWhenModelParametersDoesNotChange)

	return NewAgglomerativeHierarchicalModel

end

function AgglomerativeHierarchicalModel:setParameters(numberOfCentroids, distanceFunction, linkageFunction, highestCost, lowestCost, stopWhenModelParametersDoesNotChange)

	self.highestCost = highestCost or self.highestCost

	self.lowestCost = lowestCost or self.lowestCost

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.linkageFunction = linkageFunction or self.linkageFunction

	self.numberOfCentroids = numberOfCentroids or self.numberOfCentroids

	self.stopWhenModelParametersDoesNotChange =  self:getBooleanOrDefaultOption(stopWhenModelParametersDoesNotChange, self.stopWhenModelParametersDoesNotChange)

end

function AgglomerativeHierarchicalModel:train(featureMatrix)

	local centroidIndex1

	local centroidIndex2

	local minimumDistance

	local isOutsideCostBounds

	local numberOfIterations = 0

	local cost = 0

	local costArray = {}

	local PreviousModelParameters

	local centroidDistanceMatrix

	local centroidIndex1

	local centroidIndex2

	local areModelParametersEqual = false

	local centroids = AqwamMatrixLibrary:copy(featureMatrix)

	local newcentroid

	if self.ModelParameters then

		if (#featureMatrix[1] ~= #self.ModelParameters[1]) then error("The number of features are not the same as the model parameters!") end

		centroids = AqwamMatrixLibrary:verticalConcatenate(centroids, self.ModelParameters)

	end

	centroidDistanceMatrix = createCentroidDistanceMatrix(centroids, self.distanceFunction)

	repeat
		
		numberOfIterations += 1

		self:iterationWait()
		
		cost = self:getCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(centroids, featureMatrix, self.distanceFunction)
			
		end)
		
		if cost then
			
			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
			isOutsideCostBounds = (cost <= self.lowestCost) or (cost >= self.highestCost)
			
			if isOutsideCostBounds then break end
			
		end

		centroidIndex1, centroidIndex2 = findClosestcentroids(centroidDistanceMatrix)

		centroids = createNewcentroids(centroids, centroidIndex1, centroidIndex2)

		centroidDistanceMatrix = updateDistanceMatrix(self.linkageFunction, centroids, centroidDistanceMatrix, centroidIndex1, centroidIndex2)

		self.ModelParameters = centroids

		areModelParametersEqual = areModelParametersMatricesEqualInSizeAndValues(self.ModelParameters, PreviousModelParameters)

		

		PreviousModelParameters = self.ModelParameters

	until (#centroids == self.numberOfcentroids) or (#centroids == 1) or (areModelParametersEqual and self.stopWhenModelParametersDoesNotChange)

	self.ModelParameters = centroids

	return costArray

end

function AgglomerativeHierarchicalModel:predict(featureMatrix)

	local distance

	local closestcentroid

	local centroidVector

	local minimumDistance = math.huge

	for i, centroid in ipairs(self.ModelParameters) do

		centroidVector = {centroid}

		distance = calculateDistance(featureMatrix, centroidVector, self.distanceFunction)

		if (distance < minimumDistance) then

			minimumDistance = distance

			closestcentroid = i

		end

	end

	return closestcentroid, minimumDistance

end

return AgglomerativeHierarchicalModel
