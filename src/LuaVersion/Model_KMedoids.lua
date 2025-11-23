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

local KMedoidsModel = {}

KMedoidsModel.__index = KMedoidsModel

setmetatable(KMedoidsModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = math.huge

local defaultNumberOfClusters = 1

local defaultDistanceFunction = "Manhattan"

local defaultSetTheCentroidsDistanceFarthest = true

local function checkIfTheDataPointClusterNumberBelongsToTheCluster(dataPointClusterNumber, cluster)
	
	if (dataPointClusterNumber == cluster) then
		
		return 1
		
	else
		
		return 0
		
	end
	
end

local function createDistanceMatrix(distanceFunction, featureMatrix, modelParameters)

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
	
	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix, featureMatrix)
	
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

local function calculateCost(distanceMatrix)
	
	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix)
	
	local costMatrix = AqwamTensorLibrary:multiply(distanceMatrix, clusterAssignmentMatrix)
	
	local cost = AqwamTensorLibrary:sum(costMatrix)
	
	return cost
	
end

local function initializeCentroids(featureMatrix, numberOfClusters, distanceFunction, setTheCentroidsDistanceFarthest)

	if (setTheCentroidsDistanceFarthest) and (#featureMatrix >= numberOfClusters) then

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
	
	local ModelParameters = self.ModelParameters
	
	local numberOfClusters = self.numberOfClusters
	
	local distanceFunction = self.distanceFunction
	
	local distanceFunctionToApply = distanceFunctionDictionary[distanceFunction]

	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	local medoidMatrix = ModelParameters
	
	if (medoidMatrix) then
		
		if (#featureMatrix[1] ~= #medoidMatrix[1]) then error("The number of features are not the same as the model parameters.") end
		
	else
		
		medoidMatrix = initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply, self.setTheCentroidsDistanceFarthest)
		
	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, medoidMatrix)
	
	local oldColumnDistanceArray = {}
	
	local costArray = {}

	local numberOfIterations = 0

	local previousCost = calculateCost(distanceMatrix)
	
	local currentCost
	
	local candidateMedoidVector
	
	repeat
		
		self:iterationWait()
		
		for candidateMedoidIndex, unwrappedCandidateMedoidVector in ipairs(featureMatrix) do
			
			self:dataWait()
			
			candidateMedoidVector = {unwrappedCandidateMedoidVector}
			
			for medoidIndex, unwrappedMedoidVector in ipairs(medoidMatrix) do

				medoidMatrix[medoidIndex] = unwrappedCandidateMedoidVector
				
				for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
					
					oldColumnDistanceArray[dataIndex] = distanceMatrix[dataIndex][medoidIndex]
					
					distanceMatrix[dataIndex][medoidIndex] = distanceFunctionToApply({unwrappedFeatureVector}, candidateMedoidVector)
					
				end

				currentCost = calculateCost(distanceMatrix)

				if (currentCost > previousCost) then
					
					for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do 
						
						distanceMatrix[dataIndex][medoidIndex] = oldColumnDistanceArray[dataIndex]
						
					end

					medoidMatrix[medoidIndex] = unwrappedMedoidVector

					currentCost = previousCost
					
				else
					
					previousCost = currentCost

				end

				numberOfIterations = numberOfIterations + 1

				table.insert(costArray, currentCost)

				self:printNumberOfIterationsAndCost(numberOfIterations, currentCost)

				if (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(currentCost) or self:checkIfConverged(currentCost) then break end
				
			end

			if (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(currentCost) or self:checkIfConverged(currentCost) then break end
			
		end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(currentCost) or self:checkIfConverged(currentCost)
	
	if (self.isOutputPrinted) then

		if (currentCost == math.huge) then warn("The model diverged.") end

		if (currentCost ~= currentCost) then warn("The model produced nan (not a number) values.") end

	end
	
	self.ModelParameters = medoidMatrix
	
	return costArray
	
end

function KMedoidsModel:predict(featureMatrix, returnOriginalOutput)
	
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

return KMedoidsModel
