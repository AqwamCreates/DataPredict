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

MeanShiftModel = {}

MeanShiftModel.__index = MeanShiftModel

setmetatable(MeanShiftModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultNumberOfClusters = 0

local defaultBandwidth = 10

local defaultMode = "Hybrid"

local defaultDistanceFunction = "Euclidean"

local defaultKernelFunction = "Gaussian"

local defaultLambda = 50

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

local kernelFunctionList = {

	["Gaussian"] = function(x, kernelParameters)
		
		local denominator = 2 * math.pow(kernelParameters.bandwidth, 2)
		
		local zValue = -(x / denominator)
		
		return math.exp(zValue)

	end,
	
	["Flat"] = function(x, kernelParameters)
		
		return ((x <= kernelParameters.lambda) and 1) or 0

	end

}

local function calculateDistance(distanceFunction, vector1, vector2)
	
	return distanceFunction(vector1, vector2) 
	
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

local function createDistanceMatrix(featureMatrix, modelParameters, distanceFunction)

	local numberOfData = #featureMatrix

	local numberOfClusters = #modelParameters

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClusters}, 0)

	for datasetIndex = 1, #featureMatrix, 1 do

		for cluster = 1, #modelParameters, 1 do

			distanceMatrix[datasetIndex][cluster] = distanceFunction({featureMatrix[datasetIndex]}, {modelParameters[cluster]})

		end

	end

	return distanceMatrix

end

local function createClusterAssignmentMatrix(distanceMatrix) -- contains values of 0 and 1, where 0 is "does not belong to this cluster"

	local numberOfData = #distanceMatrix -- Number of rows

	local numberOfClusters = #distanceMatrix[1]

	local clusterAssignmentMatrix = AqwamTensorLibrary:createTensor({#distanceMatrix, #distanceMatrix[1]}, 0)

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

local function calculateCost(distanceMatrix, clusterAssignmentMatrix)

	local costMatrix = AqwamTensorLibrary:multiply(distanceMatrix, clusterAssignmentMatrix)

	local cost = AqwamTensorLibrary:sum(costMatrix)

	return cost

end

local function findEqualRowIndex(matrix1, matrix2)
	
	local index
	
	for i = 1, #matrix1, 1 do
		
		local matrixInTable = {matrix1[i]}
		
		if AqwamTensorLibrary:areMatricesEqual(matrixInTable, matrix2) then
			
			index = i
			
			break
		
		end
		
	end
	
	return index
	
end

local function calculateCentroidAndSumKernelMatrices(featureMatrix, centroidMatrix, clusterAssignmentMatrix, distanceMatrix, bandwidth, kernelFunction, kernelParameters, sumKernelMatrix, sumMultipliedKernelMatrix)
	
	for dataIndex, featureVector in ipairs(featureMatrix) do

		for clusterIndex, clusterVector in ipairs(centroidMatrix) do

			if (clusterAssignmentMatrix[dataIndex][clusterIndex] == 1) then

				local featureVector = {featureVector}

				local kernelInput = distanceMatrix[dataIndex][clusterIndex] / bandwidth

				local squaredKernelInput = math.pow(kernelInput, 2)

				local kernelVector = kernelFunction(squaredKernelInput, kernelParameters)

				local multipliedKernelVector = AqwamTensorLibrary:multiply(kernelVector, featureVector)

				local sumKernelVector = {sumKernelMatrix[clusterIndex]}

				local sumMultipliedKernelVector = {sumMultipliedKernelMatrix[clusterIndex]}

				sumKernelVector = AqwamTensorLibrary:add(sumKernelVector, kernelVector) 

				sumMultipliedKernelVector = AqwamTensorLibrary:add(sumMultipliedKernelVector, multipliedKernelVector)

				sumKernelMatrix[clusterIndex] = sumKernelVector[1]

				sumMultipliedKernelMatrix[clusterIndex] = sumMultipliedKernelVector[1]

			end

		end

	end
	
	local centroidMatrix = AqwamTensorLibrary:divide(sumMultipliedKernelMatrix, sumKernelMatrix)
	
	return centroidMatrix, sumKernelMatrix, sumMultipliedKernelMatrix
	
end

local function mergeCentroids(centroidMatrix, bandwidth, distanceFunction, sumKernelMatrix, sumMultipliedKernelMatrix)
	
	local distanceMatrix = createDistanceMatrix(centroidMatrix, centroidMatrix, distanceFunction)

	local centroidMergeArrayArray = {}
	
	local centroidMergeArray
	
	local needToBeMerged
	
	for primaryCentroidIndex, distanceVector in ipairs(distanceMatrix) do
		
		centroidMergeArray = {} 
		
		for secondaryCentroidIndex, distance in ipairs(distanceVector) do
			
			needToBeMerged = (primaryCentroidIndex ~= secondaryCentroidIndex) and (distance <= bandwidth)
			
			if (needToBeMerged) then table.insert(centroidMergeArray, secondaryCentroidIndex) end
			
		end
		
		centroidMergeArrayArray[primaryCentroidIndex] = centroidMergeArray
		
	end
	
	local mergedFlagArray = {}
	
	local newCentroidMatrix = {}
	
	local newSumKernelMatrix = {}
	
	local newSumMultipliedKernelMatrix = {}
	
	local numberOfFeatures = #centroidMatrix[1]

	for i, mergeArray in ipairs(centroidMergeArrayArray) do
		
		if (not mergedFlagArray[i]) then
			
			local combinedIndices = {i}
			
			for _, j in ipairs(mergeArray) do
				
				if (not mergedFlagArray[j]) then table.insert(combinedIndices, j) end
				
			end

			local newSumKernelVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 0)
			
			local newSumMultipliedKernelVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 0)

			for _, idx in ipairs(combinedIndices) do
				
				newSumKernelVector = AqwamTensorLibrary:add(newSumKernelVector, {sumKernelMatrix[idx]})
				
				newSumMultipliedKernelVector = AqwamTensorLibrary:add(newSumMultipliedKernelVector, {sumMultipliedKernelMatrix[idx]})
				
				mergedFlagArray[idx] = true
				
			end

			local newCentroidVector = AqwamTensorLibrary:divide(newSumMultipliedKernelVector, newSumKernelVector)
			
			table.insert(newCentroidMatrix, newCentroidVector[1])
			
			table.insert(newSumKernelMatrix, newSumKernelVector[1])
			
			table.insert(newSumMultipliedKernelMatrix, newSumMultipliedKernelVector[1])
			
		end
		
	end

	return newCentroidMatrix, newSumKernelMatrix, newSumMultipliedKernelMatrix
	
end

function MeanShiftModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewMeanShiftModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewMeanShiftModel, MeanShiftModel)
	
	NewMeanShiftModel:setName("MeanShift")
	
	local bandwidth = parameterDictionary.bandwidth or defaultBandwidth
	
	NewMeanShiftModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters

	NewMeanShiftModel.bandwidth = bandwidth
	
	NewMeanShiftModel.mode = parameterDictionary.mode or defaultMode
	
	NewMeanShiftModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewMeanShiftModel.kernelFunction = parameterDictionary.kernelFunction or defaultKernelFunction
	
	local kernelParameters = {
		
		bandwidth = bandwidth,
		
		lambda = parameterDictionary.lambda or defaultLambda,
		
	}
	
	NewMeanShiftModel.kernelParameters = kernelParameters
	
	return NewMeanShiftModel
	
end

function MeanShiftModel:train(featureMatrix)
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local bandwidth = self.bandwidth
	
	local mode = self.mode
	
	local distanceFunction = self.distanceFunction
	
	local kernelFunction = self.kernelFunction
	
	local kernelParameters = self.kernelParameters
	
	local ModelParameters = self.ModelParameters or {}
	
	local centroidMatrix = ModelParameters[1]
	
	local sumKernelMatrix = ModelParameters[2]
	
	local sumMultipliedKernelMatrix = ModelParameters[3]
	
	if (mode == "Hybrid") then

		mode = (centroidMatrix and sumKernelMatrix and sumMultipliedKernelMatrix and "Sequential") or "Batch"		

	end

	local distanceFunctionToApply = distanceFunctionList[distanceFunction]

	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	local kernelFunctionToApply = kernelFunctionList[kernelFunction]
	
	if (not kernelFunctionToApply) then error("Unknown kernel function.") end
	
	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]
	
	local costArray = {}

	local numberOfIterations = 0
	
	local centroidDimensionSizeArray

	local distanceMatrix

	local clusterAssignmentMatrix 

	local cost
	
	-- This is batch and not sequential mode. So, we need to reset the whole model to get fresh model parameters.
	
	if (mode == "Batch") then 

		centroidMatrix = nil
		
		sumKernelMatrix = nil
		
		sumMultipliedKernelMatrix = nil

	end
	
	-- The noise is added to the feature matrix is because we want to avoid the cost to be zero at the first iteration.

	centroidMatrix = centroidMatrix or AqwamTensorLibrary:add(featureMatrix, AqwamTensorLibrary:createRandomUniformTensor({numberOfData, numberOfFeatures}), -1e-16, 1e-16)
	
	centroidDimensionSizeArray = {#centroidMatrix, numberOfFeatures}

	sumKernelMatrix = sumKernelMatrix or AqwamTensorLibrary:createTensor(centroidDimensionSizeArray)

	sumMultipliedKernelMatrix = sumMultipliedKernelMatrix or AqwamTensorLibrary:createTensor(centroidDimensionSizeArray)

	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, distanceFunctionToApply)
		
		clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix)
		
		centroidMatrix, sumKernelMatrix, sumMultipliedKernelMatrix = calculateCentroidAndSumKernelMatrices(featureMatrix, centroidMatrix, clusterAssignmentMatrix, distanceMatrix, bandwidth, kernelFunctionToApply, kernelParameters, sumKernelMatrix, sumMultipliedKernelMatrix)
		
		centroidMatrix, sumKernelMatrix, sumMultipliedKernelMatrix = mergeCentroids(centroidMatrix, bandwidth, distanceFunctionToApply, sumKernelMatrix, sumMultipliedKernelMatrix)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(distanceMatrix, clusterAssignmentMatrix)
			
		end)
		
		if (cost) then
			
			table.insert(costArray, cost)
			
			self:printNumberOfIterationsAndCost(numberOfIterations, cost)
			
		end
		
	until (numberOfIterations == maximumNumberOfIterations) or (#centroidMatrix <= numberOfClusters) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.ModelParameters = {centroidMatrix, sumKernelMatrix, sumMultipliedKernelMatrix}
	
	return costArray
	
end

function MeanShiftModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceFunctionToApply = distanceFunctionList[self.distanceFunction]
	
	local ModelParameters = self.ModelParameters
	
	local centroidMatrix = ModelParameters[1]
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, distanceFunctionToApply)
	
	if (returnOriginalOutput) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)
	
	return clusterNumberVector, clusterDistanceVector
	
end

return MeanShiftModel
