--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

MeanShiftModel = {}

MeanShiftModel.__index = MeanShiftModel

setmetatable(MeanShiftModel, BaseModel)

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaximumNumberOfIterations = 500

local defaultBandwidth = math.huge

local defaultBandwidthStep = 100

local defaultDistanceFunction = "Euclidean"

local defaultKernelFunction = "Gaussian"

local defaultLambda = 50

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
	
	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamMatrixLibrary:dotProduct(x1, AqwamMatrixLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamMatrixLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamMatrixLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamMatrixLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamMatrixLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2, 2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

local kernelFunctionList = {

	["Gaussian"] = function(featureMatrix, kernelParameters)
		
		local functionToApply = function(x) return math.exp(-0.5 * math.pow(x, 2)) / math.sqrt(2 * math.pi) end
		
		return AqwamMatrixLibrary:applyFunction(functionToApply, featureMatrix)

	end,
	
	["Flat"] = function(featureMatrix, kernelParameters)
		
		local lambda = kernelParameters.lambda or defaultLambda
		
		local functionToApply = function(x) return ((x <= lambda) and 1) or 0 end
		
		return AqwamMatrixLibrary:applyFunction(functionToApply, featureMatrix)

	end

}

local function calculateDistance(vector1, vector2, distanceFunction)
	
	return distanceFunctionList[distanceFunction](vector1, vector2) 
	
end

local function assignToCluster(distanceMatrix) -- Number of columns -> number of clusters
	
	local clusterNumberVector = AqwamMatrixLibrary:createMatrix(#distanceMatrix, 1)
	
	local clusterDistanceVector = AqwamMatrixLibrary:createMatrix(#distanceMatrix, 1) 
	
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

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfClusters)

	for datasetIndex = 1, #featureMatrix, 1 do

		for cluster = 1, #modelParameters, 1 do

			distanceMatrix[datasetIndex][cluster] = calculateDistance({featureMatrix[datasetIndex]}, {modelParameters[cluster]} , distanceFunction)

		end

	end

	return distanceMatrix

end

local function createClusterAssignmentMatrix(distanceMatrix, bandwidth) -- contains values of 0 and 1, where 0 is "does not belong to this cluster"

	local numberOfData = #distanceMatrix -- Number of rows

	local numberOfClusters = #distanceMatrix[1]

	local clusterAssignmentMatrix = AqwamMatrixLibrary:createMatrix(#distanceMatrix, #distanceMatrix[1])

	local dataPointClusterNumber

	for dataIndex = 1, numberOfData, 1 do
		
		for clusterIndex = 1, numberOfClusters, 1 do
			
			if (distanceMatrix[dataIndex][clusterIndex] > bandwidth) then continue end
			
			clusterAssignmentMatrix[dataIndex][clusterIndex] = 1
				
		end

	end

	return clusterAssignmentMatrix

end

local function calculateCost(featureMatrix, modelParameters, distanceFunction)
	
	local cost = 0
	
	for i = 1, #featureMatrix do
		
		local minimumDistance = math.huge
		
		for j = 1, #modelParameters do
			
			local distance = calculateDistance({featureMatrix[i]}, {modelParameters[j]}, distanceFunction)
		
			minimumDistance = math.min(minimumDistance, distance)
			
		end
		
		cost = cost + minimumDistance
		
	end
	
	return cost
	
end

local function findEqualRowIndex(matrix1, matrix2)
	
	local index
	
	for i = 1, #matrix1, 1 do
		
		local matrixInTable = {matrix1[i]}
		
		if AqwamMatrixLibrary:areMatricesEqual(matrixInTable, matrix2) then
			
			index = i
			
			break
		
		end
		
	end
	
	return index
	
end

local function removeDuplicateRows(ModelParameters)
	
	local UniqueModelParameters = {}
	
	for i = 1, #ModelParameters, 1 do
		
		local index = findEqualRowIndex(UniqueModelParameters, {ModelParameters[i]})
		
		if (index == nil) then table.insert(UniqueModelParameters, ModelParameters[i]) end
		
	end
	
	return UniqueModelParameters
	
end

local function createWeightedMeanMatrix(featureMatrix, ModelParameters, clusterAssignmentMatrix, bandwidth, kernelFunction)
	
	local numberOfData = #featureMatrix
	
	local numberOfClusters = #ModelParameters
	
	local selectedKernelFunction = kernelFunctionList[kernelFunction]
	
	local sumKernelVector = AqwamMatrixLibrary:createMatrix(#ModelParameters, #ModelParameters[1])
	
	local sumMultipliedKernelVector = AqwamMatrixLibrary:createMatrix(#ModelParameters, #ModelParameters[1])
	
	for dataIndex, featureVector in ipairs(featureMatrix) do
		
		for clusterIndex, clusterVector in ipairs(ModelParameters) do
			
			if (clusterAssignmentMatrix[dataIndex][clusterIndex] ~= 1) then continue end
			
			local featureVector = {featureVector}

			local subtractedVector = AqwamMatrixLibrary:subtract(featureVector, {clusterVector})
			
			local kernelInputVector = AqwamMatrixLibrary:divide(subtractedVector, bandwidth)
			
			local kernelVector = selectedKernelFunction(subtractedVector)
			
			local multipliedKernelVector = AqwamMatrixLibrary:multiply(kernelVector, featureVector)
			
			local kernelizedSumVector = {sumKernelVector[clusterIndex]}
			
			local multipliedKernelizedSumVector = {sumMultipliedKernelVector[clusterIndex]}
			
			kernelizedSumVector = AqwamMatrixLibrary:add(kernelizedSumVector, kernelVector) 
			
			multipliedKernelizedSumVector = AqwamMatrixLibrary:add(multipliedKernelizedSumVector, multipliedKernelVector)

			sumKernelVector[clusterIndex] = kernelizedSumVector[1]
			
			sumMultipliedKernelVector[clusterIndex] = multipliedKernelizedSumVector[1]
			
		end
		
	end
	
	local weightedMeanMatrix = AqwamMatrixLibrary:divide(sumMultipliedKernelVector, sumKernelVector)
	
	return weightedMeanMatrix
	
end

function MeanShiftModel.new(maximumNumberOfIterations, bandwidth, distanceFunction, kernelFunction)
	
	local NewMeanShiftModel = BaseModel.new()
	
	setmetatable(NewMeanShiftModel, MeanShiftModel)
	
	NewMeanShiftModel.maximumNumberOfIterations = maximumNumberOfIterations or defaultMaximumNumberOfIterations

	NewMeanShiftModel.bandwidth = bandwidth or defaultBandwidth
	
	NewMeanShiftModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	NewMeanShiftModel.kernelFunction = kernelFunction or defaultKernelFunction
	
	return NewMeanShiftModel
	
end

function MeanShiftModel:setParameters(maximumNumberOfIterations, bandwidth, distanceFunction, kernelFunction)
	
	self.maximumNumberOfIterations = maximumNumberOfIterations or self.maximumNumberOfIterations

	self.bandwidth = bandwidth or self.bandwidth
	
	self.distanceFunction = distanceFunction or self.distanceFunction
	
	self.kernelFunction = kernelFunction or self.kernelFunction
	
end

function MeanShiftModel:train(featureMatrix)
	
	local isOutsideCostBounds
	
	local PreviousModelParameters
	
	local cost
	
	local costArray = {}
	
	local weights = {}
	
	local numberOfIterations = 0
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local bandwidth = self.bandwidth
	
	local distanceFunction = self.distanceFunction
	
	local kernelFunction = self.kernelFunction
	
	local ModelParameters = self.ModelParameters
		
	if (not ModelParameters) then
		
		ModelParameters = featureMatrix
		
	end
	
	repeat
		
		numberOfIterations += 1
		
		self:iterationWait()

		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(featureMatrix, ModelParameters, distanceFunction)
			
		end)
		
		if cost then
			
			table.insert(costArray, cost)
			
			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
		end
		
		local distanceMatrix = createDistanceMatrix(featureMatrix, ModelParameters, distanceFunction)
		
		local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix, bandwidth)

		local weightedMeanMatrix = createWeightedMeanMatrix(featureMatrix, ModelParameters, clusterAssignmentMatrix, bandwidth, kernelFunction)
		
		ModelParameters = AqwamMatrixLibrary:subtract(ModelParameters, weightedMeanMatrix)
		
		ModelParameters = removeDuplicateRows(ModelParameters)
		
		self.ModelParameters = ModelParameters
		
	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	return costArray
	
end

function MeanShiftModel:predict(featureMatrix, returnOriginalOutput)
	
	local distanceMatrix = createDistanceMatrix(self.ModelParameters, featureMatrix, self.distanceFunction)
	
	if (returnOriginalOutput == true) then return distanceMatrix end

	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)
	
	return clusterNumberVector, clusterDistanceVector
	
end

return MeanShiftModel
