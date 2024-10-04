--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseModel = require(script.Parent.BaseModel)

MeanShiftModel = {}

MeanShiftModel.__index = MeanShiftModel

setmetatable(MeanShiftModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaximumNumberOfIterations = 500

local defaultBandwidth = 100

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

	["Gaussian"] = function(x, kernelParameters)
		
		return math.exp(-0.5 * math.pow(x, 2)) / math.sqrt(2 * math.pi)

	end,
	
	["Flat"] = function(x, kernelParameters)
		
		local lambda = kernelParameters.lambda or defaultLambda
		
		return ((x <= lambda) and 1) or 0

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

local function createClusterAssignmentMatrix(distanceMatrix) -- contains values of 0 and 1, where 0 is "does not belong to this cluster"

	local numberOfData = #distanceMatrix -- Number of rows

	local numberOfClusters = #distanceMatrix[1]

	local clusterAssignmentMatrix = AqwamMatrixLibrary:createMatrix(#distanceMatrix, #distanceMatrix[1])

	local dataPointClusterNumber

	for dataIndex = 1, numberOfData, 1 do

		local distanceVector = {distanceMatrix[dataIndex]}

		local _, vectorIndexArray = AqwamMatrixLibrary:findMinimumValue(distanceVector)

		if (vectorIndexArray == nil) then continue end

		local clusterNumber = vectorIndexArray[2]

		clusterAssignmentMatrix[dataIndex][clusterNumber] = 1

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

local function createWeightedMeanMatrix(featureMatrix, ModelParameters, bandwidth, distanceFunction, kernelFunction, kernelParameters)
	
	local numberOfData = #featureMatrix
	
	local numberOfClusters = #ModelParameters
	
	local selectedKernelFunction = kernelFunctionList[kernelFunction]
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, ModelParameters, distanceFunction)

	local clusterAssignmentMatrix = createClusterAssignmentMatrix(distanceMatrix)
	
	local sumKernelMatrix = AqwamMatrixLibrary:createMatrix(#ModelParameters, #ModelParameters[1])
	
	local sumMultipliedKernelMatrix = AqwamMatrixLibrary:createMatrix(#ModelParameters, #ModelParameters[1])
	
	for dataIndex, featureVector in ipairs(featureMatrix) do
		
		for clusterIndex, clusterVector in ipairs(ModelParameters) do
			
			if (clusterAssignmentMatrix[dataIndex][clusterIndex] ~= 1) then continue end
			
			local featureVector = {featureVector}
			
			local kernelInput = distanceMatrix[dataIndex][clusterIndex] / bandwidth
			
			local squaredKernelInput = math.pow(kernelInput, 2)
			
			local kernelVector = selectedKernelFunction(squaredKernelInput, kernelParameters)
			
			local multipliedKernelVector = AqwamMatrixLibrary:multiply(kernelVector, featureVector)
			
			local sumKernelVector = {sumKernelMatrix[clusterIndex]}
			
			local sumMultipliedKernelVector = {sumMultipliedKernelMatrix[clusterIndex]}
			
			sumKernelVector = AqwamMatrixLibrary:add(sumKernelVector, kernelVector) 
			
			sumMultipliedKernelVector = AqwamMatrixLibrary:add(sumMultipliedKernelVector, multipliedKernelVector)

			sumKernelMatrix[clusterIndex] = sumKernelVector[1]
			
			sumMultipliedKernelMatrix[clusterIndex] = sumMultipliedKernelVector[1]
			
		end
		
	end
	
	local weightedMeanMatrix = AqwamMatrixLibrary:divide(sumMultipliedKernelMatrix, sumKernelMatrix)
	
	return weightedMeanMatrix
	
end

function MeanShiftModel.new(maximumNumberOfIterations, bandwidth, distanceFunction, kernelFunction, kernelParameters)
	
	local NewMeanShiftModel = BaseModel.new()
	
	setmetatable(NewMeanShiftModel, MeanShiftModel)
	
	NewMeanShiftModel.maximumNumberOfIterations = maximumNumberOfIterations or defaultMaximumNumberOfIterations

	NewMeanShiftModel.bandwidth = bandwidth or defaultBandwidth
	
	NewMeanShiftModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	NewMeanShiftModel.kernelFunction = kernelFunction or defaultKernelFunction
	
	NewMeanShiftModel.kernelParameters = kernelParameters or {}
	
	return NewMeanShiftModel
	
end

function MeanShiftModel:setParameters(maximumNumberOfIterations, bandwidth, distanceFunction, kernelFunction, kernelParameters)
	
	self.maximumNumberOfIterations = maximumNumberOfIterations or self.maximumNumberOfIterations

	self.bandwidth = bandwidth or self.bandwidth
	
	self.distanceFunction = distanceFunction or self.distanceFunction
	
	self.kernelFunction = kernelFunction or self.kernelFunction
	
	self.kernelParameters = kernelParameters or self.kernelParameters
	
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
	
	local kernelParameters = self.kernelParameters
	
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
		
		ModelParameters = createWeightedMeanMatrix(featureMatrix, ModelParameters, bandwidth, distanceFunction, kernelFunction, kernelParameters)
		
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