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

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

BisectingClusterModel = {}

BisectingClusterModel.__index = BisectingClusterModel

setmetatable(BisectingClusterModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultNumberOfClusters = 2

local defaultDistanceFunction = "Euclidean"

local defaultSplitCriterion = "LargestCluster"

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

local function extractSubsetMatrix(matrix, indexArray)
	
	local subMatrix = {}
	
	local index = 1
	
	for _, clusterIndex in ipairs(indexArray) do
		
		subMatrix[index] = matrix[clusterIndex]
		
		index = index + 1
		
	end
	
	return subMatrix
	
end

local function calculateCentroid(featureMatrix, indexArray)
	
	local numberOfIndices = #indexArray
	
	if (numberOfIndices == 0) then return nil end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local unwrappedCentroidVector = table.create(numberOfFeatures, 0)
	
	local unwrappedFeatureVector
	
	for _, index in ipairs(indexArray) do
		
		unwrappedFeatureVector = featureMatrix[index]
		
		for featureIndex, value in ipairs(unwrappedFeatureVector) do
			
			unwrappedCentroidVector[featureIndex] = unwrappedCentroidVector[featureIndex] + value
			
		end
		
	end
	
	for featureIndex, value in ipairs(unwrappedCentroidVector) do
		
		unwrappedCentroidVector[featureIndex] = value / numberOfIndices
		
	end
		
	return unwrappedCentroidVector
		
end

local function calculateSumOfSquaredError(featureMatrix, indexArray, centroidVector, distanceFunction)

	local sumOfSquaredError = 0
	
	if (#indexArray == 0) then return sumOfSquaredError end
	
	local featureVector
	
	local distance
	
	for _, index in ipairs(indexArray) do
		
		featureVector = {featureMatrix[index]}
		
		distance = distanceFunction(featureVector, centroidVector)

		sumOfSquaredError = sumOfSquaredError + math.pow(distance)
		
	end
	
	return sumOfSquaredError
	
end

function BisectingClusterModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewBisectingClusterModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewBisectingClusterModel, BisectingClusterModel)
	
	NewBisectingClusterModel:setName("BisectingCluster")
	
	NewBisectingClusterModel.Model = parameterDictionary.Model

	NewBisectingClusterModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters
	
	NewBisectingClusterModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewBisectingClusterModel.splitCriterion = parameterDictionary.splitCriterion or defaultSplitCriterion
	
	return NewBisectingClusterModel
	
end

function BisectingClusterModel:train(featureMatrix)
	
	local Model = self.Model
	
	if (not Model) then error("No model") end
	
	local numberOfClusters = self.numberOfClusters
	
	local distanceFunction = self.distanceFunction
	
	local splitCriterion = self.splitCriterion
	
	local secondaryNumberOfClusters = Model.numberOfClusters

	local numberOfIterations = 0
	
	local primaryCostArray = {}
	
	local secondaryCostArray = {}
	
	local cost

	local distanceMatrix
	
	local clusterIndexToSplit
	
	local numberOfData
	
	local clusterFeatureMatrix
	
	local maximumCriterion
	
	local criterion
	
	local clusterInformationDictionary
	
	local clusterIndexVector
	
	local sumOfSquaredErrorValue
	
	local dataIndexArray
	
	local leftDataIndexArray
	
	local rightDataIndexArray
	
	local targetDataIndexArray
	
	local clusterIndex
	
	local unwrappedCentroidVector
	
	local clusterInformationDictionaryArray
	
	dataIndexArray = {}
	
	for i, _ in ipairs(featureMatrix) do dataIndexArray[i] = i end
	
	unwrappedCentroidVector = calculateCentroid(featureMatrix, dataIndexArray)
	
	sumOfSquaredErrorValue = calculateSumOfSquaredError(featureMatrix, dataIndexArray, unwrappedCentroidVector, distanceFunction)
	
	clusterInformationDictionaryArray = {{dataIndexArray = dataIndexArray, sumOfSquaredErrorValue = sumOfSquaredErrorValue}}
	
	Model.numberOfClusters = 2
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		clusterIndexToSplit = 1
		
		maximumCriterion = -math.huge
		
		for i, clusterInformationDictionary in ipairs(clusterInformationDictionaryArray) do
			
			dataIndexArray = clusterInformationDictionary.dataIndexArray
			
			numberOfData = #dataIndexArray
			
			criterion = (splitCriterion == "LargestCluster") and numberOfData or clusterInformationDictionary.sumOfSquaredErrorValue
			
			if (criterion > maximumCriterion) and (numberOfData ~= 1) then
				
				maximumCriterion = criterion
				
				clusterIndexToSplit = i
				
			end
			
		end
		
		clusterInformationDictionary = clusterInformationDictionaryArray[clusterIndexToSplit]
		
		dataIndexArray = clusterInformationDictionary.dataIndexArray
		
		clusterFeatureMatrix = extractSubsetMatrix(featureMatrix, dataIndexArray)
		
		secondaryCostArray = Model:train(clusterFeatureMatrix)
		
		clusterIndexVector = Model:predict(clusterFeatureMatrix)
		
		leftDataIndexArray = {}
		
		rightDataIndexArray = {}
		
		for dataIndex, unwrappedClusterIndex in ipairs(clusterIndexVector) do
			
			clusterIndex = unwrappedClusterIndex[1]
			
			targetDataIndexArray = ((clusterIndex == 1) and leftDataIndexArray) or rightDataIndexArray
			
			table.insert(targetDataIndexArray, dataIndexArray[dataIndex])
			
		end
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return secondaryCostArray[#secondaryCostArray]

		end)
		
		if (cost) then

			table.insert(primaryCostArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
		table.remove(clusterInformationDictionaryArray, clusterIndexToSplit)
		
		table.insert(clusterInformationDictionaryArray, {dataIndexArray = leftDataIndexArray, sumOfSquaredErrorValue = 0})
		
		table.insert(clusterInformationDictionaryArray, {dataIndexArray = rightDataIndexArray, sumOfSquaredErrorValue = 0})
		
	until (#clusterInformationDictionaryArray == numberOfClusters) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	self.ModelParameters = centroidMatrix
	
	Model.numberOfClusters = secondaryNumberOfClusters
	
	return primaryCostArray
	
end

function BisectingClusterModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceFunctionToApply = distanceFunctionDictionary[self.distanceFunction]
	
	local centroidMatrix = self.ModelParameters
	
	if (not centroidMatrix) then
		
		local numberOfClusters = self.numberOfClusters
		
		if (numberOfClusters == 1) then
			
			centroidMatrix = AqwamTensorLibrary:mean(featureMatrix, 1)
			
		else
			
			centroidMatrix = self:initializeMatrixBasedOnMode({numberOfClusters, #featureMatrix[1]})
			
		end

		centroidMatrix = self:initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply)
		
		self.ModelParameters = centroidMatrix

	end
	
	local distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, centroidMatrix)
	
	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return BisectingClusterModel
