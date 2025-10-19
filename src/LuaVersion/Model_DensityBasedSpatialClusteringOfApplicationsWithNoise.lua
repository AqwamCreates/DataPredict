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

DensityBasedSpatialClusteringOfApplicationsWithNoiseModel = {}

DensityBasedSpatialClusteringOfApplicationsWithNoiseModel.__index = DensityBasedSpatialClusteringOfApplicationsWithNoiseModel

setmetatable(DensityBasedSpatialClusteringOfApplicationsWithNoiseModel, IterativeMethodBaseModel)

local defaultMinimumNumberOfPoints = 2

local defaultDistanceFunction = "Manhattan"

local defaultEpsilon = 10


local function getNeighbors(currentCorePointNumber, featureMatrix, epsilon, distanceFunction)
	
	local distance
	
	local neighborArray = {}
	
	for i = 1, #featureMatrix, 1 do
		
		if (i ~= currentCorePointNumber) then
			
			distance = distanceFunction({featureMatrix[currentCorePointNumber]}, {featureMatrix[i]})
			
			if (distance <= epsilon) then table.insert(neighborArray, i) end
			
		end
		
	end
	
	return neighborArray
	
end

local function mergeTables(table1, table2)
	
	for _, value in ipairs(table2) do table.insert(table1, value) end
	
	return table1
	
end

local function expandCluster(currentCorePointNumber, neighborArray, neighbouringCorePointNumber, clusterArray, hasVisitedCorePointNumberArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	clusterArray[neighbouringCorePointNumber] = clusterArray[neighbouringCorePointNumber] or {}
	
	clusterArray[neighbouringCorePointNumber][#clusterArray[neighbouringCorePointNumber] + 1] = currentCorePointNumber
	
	for _, neighbouringPointNumber in ipairs(neighborArray) do
		
		if (not hasVisitedCorePointNumberArray[neighbouringPointNumber]) then

			hasVisitedCorePointNumberArray[neighbouringPointNumber] = true

			local qNeighbors = getNeighbors(neighbouringPointNumber, featureMatrix, epsilon, distanceFunction)

			if (#qNeighbors >= minimumNumberOfPoints) then

				neighborArray = mergeTables(neighborArray, qNeighbors)

			end

		end

		local isInCluster = false
		
		for _, cluster in ipairs(clusterArray) do
			
			if (cluster[neighbouringPointNumber]) then

				isInCluster = true

				break

			end
			
		end

		if (not isInCluster) then

			clusterArray[neighbouringCorePointNumber][#clusterArray[neighbouringCorePointNumber] + 1] = neighbouringPointNumber

		end
		
	end
	
end

local function calculateCost(distanceFunction, featureMatrix, clusters)
	
	local cost = 0
	
	for cluster_id, clusterPoints in pairs(clusters) do
			
		for i = 1, #clusterPoints, 1 do
				
			for j = i + 1, #clusterPoints, 1 do
					
				cost = cost + distanceFunction({featureMatrix[clusterPoints[i]]}, {featureMatrix[clusterPoints[j]]})
					
			end
				
		end
		
	end
	
	return cost
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel, DensityBasedSpatialClusteringOfApplicationsWithNoiseModel)
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel:setName("DensityBasedSpatialClusteringOfApplicationsWithNoise")
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.minimumNumberOfPoints = parameterDictionary.minimumNumberOfPoints or defaultMinimumNumberOfPoints
	
	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	return NewDensityBasedSpatialClusteringOfApplicationsWithNoiseModel
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:train(featureMatrix)
	
	if (self.ModelParameters) then

		local storedFeatureMatrix = self.ModelParameters[1]

		if (#storedFeatureMatrix[1] ~= #featureMatrix[1]) then error("The previous and current feature matrices do not have the same number of features.") end 

		featureMatrix = AqwamTensorLibrary:concatenate(featureMatrix, storedFeatureMatrix, 1)

	end
	
	local distanceFunction = self.distanceFunction
	
	local distanceFunctionToApply = distanceFunctionDictionary[distanceFunction]
	
	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	local minimumNumberOfPoints = self.minimumNumberOfPoints

	local epsilon = self.epsilon
	
	local costArray = {}

	local clusterArray = {}

	local noiseCorePointNumberArray = {}

	local hasVisitedCorePointNumberArray = {}
	
	local cost
	
	local neighbouringCorePointNumber 
	
	local neighborArray
	
	for currentCorePointNumber, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		self:iterationWait()

		if (not hasVisitedCorePointNumberArray[currentCorePointNumber]) then

			hasVisitedCorePointNumberArray[currentCorePointNumber] = true

			neighborArray = getNeighbors(currentCorePointNumber, featureMatrix, epsilon, distanceFunctionToApply)

			if (#neighborArray < self.minimumNumberOfPoints) then

				table.insert(noiseCorePointNumberArray, currentCorePointNumber)

			else

				neighbouringCorePointNumber = #clusterArray + 1

				expandCluster(currentCorePointNumber, neighborArray, neighbouringCorePointNumber, clusterArray, hasVisitedCorePointNumberArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)

			end

		end

		cost = self:calculateCostWhenRequired(currentCorePointNumber, function()

			return calculateCost(distanceFunctionToApply, featureMatrix, clusterArray)

		end)

		if cost then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(currentCorePointNumber, cost)

			if self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) then break end

		end
		
	end
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	self.ModelParameters = {featureMatrix, clusterArray}
	
	return costArray
	
end

function DensityBasedSpatialClusteringOfApplicationsWithNoiseModel:predict(featureMatrix)
	
	local numberOfData = #featureMatrix
	
	local dimensionSizeArray = {numberOfData, 1}
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		local placeholderClusterVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, nil)
		
		local placeholderDistanceVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)
		
		return placeholderClusterVector, placeholderDistanceVector
		
	end
	
	local distanceFunctionToApply = distanceFunctionDictionary[self.distanceFunction]
	
	local shortestDistanceVector = AqwamTensorLibrary:createTensor(dimensionSizeArray)

	local closestClusterVector = AqwamTensorLibrary:createTensor(dimensionSizeArray)
	
	local storedFeatureVector, cluster = table.unpack(ModelParameters)
	
	local closestCluster
	
	local shortestDistance
	
	local featureVector
	
	for i, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		closestCluster = nil

		shortestDistance = math.huge

		featureVector = {unwrappedFeatureVector}

		for clusterNumber, clusterPoints in ipairs(cluster) do

			local distance = 0
			
			for j, pointNumber in ipairs(clusterPoints) do

				distance = distance + distanceFunctionToApply(featureVector, {storedFeatureVector[pointNumber]})
				
			end

			distance = distance / #clusterPoints

			if (distance < shortestDistance) then

				closestCluster = clusterNumber

				shortestDistance = distance

			end

		end

		closestClusterVector[i][1] = closestCluster

		shortestDistanceVector[i][1] = shortestDistance
		
	end
	
	return closestClusterVector, shortestDistanceVector
	
end

return DensityBasedSpatialClusteringOfApplicationsWithNoiseModel
