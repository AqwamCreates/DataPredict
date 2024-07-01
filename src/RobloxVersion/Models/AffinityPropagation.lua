local BaseModel = require(script.Parent.BaseModel)

local AffinityPropagationModel = {}

AffinityPropagationModel.__index = AffinityPropagationModel

setmetatable(AffinityPropagationModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultDamping = 0.5

local defaultDistanceFunction = "Euclidean"

local defaultPreferenceType = "Median"

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

local function calculateDistance(vector1, vector2, distanceFunction)

	return distanceFunctionList[distanceFunction](vector1, vector2) 

end

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamMatrixLibrary:createMatrix(numberOfData1, numberOfData2)

	for i = 1, numberOfData1, 1 do

		for j = 1, numberOfData2, 1 do

			distanceMatrix[i][j] = calculateDistance({matrix1[i]}, {matrix2[j]}, distanceFunction)

		end

	end

	return distanceMatrix

end

local function getMedian(array)
	
	table.sort(array)
	
	local mid = math.floor(#array / 2)
	
	if ((#array % 2) == 0) then
		
		return ((array[mid] + array[mid + 1]) / 2)
		
	else
		
		return array[mid + 1]
		
	end
	
end

local function getAverage(array)
	
	local total = 0
	
	for _, value in ipairs(array) do
		
		total = total + value
		
	end
	
	return (total / #array)
	
end

local function setPreferencesToSimilarityMatrix(similarityMatrix, numberOfData, preferenceType, preferenceValueArray)
	
	local preferenceValue
	
	local triangularElementArray = {} -- Collect upper triangular non-diagonal elements

	for i = 1, numberOfData, 1 do
		
		for j = i + 1, numberOfData, 1 do
			
			table.insert(triangularElementArray, similarityMatrix[i][j])
			
		end
		
	end

	if (preferenceType == "Median") then
		
		preferenceValue = getMedian(triangularElementArray)
		
	elseif (preferenceType == "Average") then

		preferenceValue = getAverage(triangularElementArray)

	elseif (preferenceType == "Minimum") then

		preferenceValue = math.min(table.unpack(triangularElementArray))
		
	elseif (preferenceType == "Maximum") then

		preferenceValue = math.max(table.unpack(triangularElementArray))
		
	elseif (preferenceType == "Precomputed") then
		
	else

		error("Invalid preference type!")

	end
	
	for i = 1, numberOfData do -- Fill diagonal with the computed preference value
		
		if (preferenceType == "Precomputed") then
			
			similarityMatrix[i][i] = preferenceValueArray[i]
			
		else
			
			similarityMatrix[i][i] = preferenceValue
			
		end
		
	end
	
	return similarityMatrix

end

local function calculateResponsibilityMatrix(responsibilityMatrix, availabilityMatrix, similarityMatrix)
	
	local numberOfData = #responsibilityMatrix

	for i = 1, numberOfData do

		for j = 1, numberOfData do

			local maxResponsibility = -math.huge

			for k = 1, numberOfData do

				if (k == j) then continue end

				maxResponsibility = math.max(maxResponsibility, similarityMatrix[i][k] + availabilityMatrix[k][j])

			end

			responsibilityMatrix[i][j] = similarityMatrix[i][j] - maxResponsibility

		end

	end
	
	return responsibilityMatrix

end

local function calculateAvailibilityMatrix(responsibilityMatrix, availibilityMatrix, damping)
	
	local updateFactor = 1 - damping
	
	local numberOfData = #responsibilityMatrix

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfData, 1 do
			
			local availability
			
			local sumMaxAvailability = 0

			if (i ~= j) then
				
				for k = 1, numberOfData, 1 do

					if (k == i) and (k == j) then continue end

					local maxAvailability = math.max(0, responsibilityMatrix[k][j])
					
					sumMaxAvailability = sumMaxAvailability + maxAvailability

				end
				
				availability = (updateFactor * (responsibilityMatrix[j][j] + sumMaxAvailability)) + (damping * availibilityMatrix[i][j])
				
				availability = math.min(0, availability)
				
			else
				
				for k = 1, numberOfData, 1 do

					if (k == i) then continue end

					local maxAvailability = math.max(0, responsibilityMatrix[k][j])

					sumMaxAvailability = sumMaxAvailability + maxAvailability

				end
				
				availability = (updateFactor * sumMaxAvailability) + (damping * availibilityMatrix[i][j])

			end
			
			availibilityMatrix[i][j] = availability

		end

	end
	
	return availibilityMatrix

end

local function calculateCost(clusters, responsibilityMatrix)

	local totalCost = 0

	for i = 1, #clusters do

		totalCost += responsibilityMatrix[i][clusters[i][1]]

	end

	return totalCost

end

local function assignClusters(responsibilityMatrix, availibilityMatrix)
	
	local calculatedValuesMatrix = AqwamMatrixLibrary:add(responsibilityMatrix, availibilityMatrix)
	
	local clusterVector = AqwamMatrixLibrary:createMatrix(#responsibilityMatrix, 1)
	
	for i = 1, #calculatedValuesMatrix, 1 do
		
		local calculatedValuesVector = {calculatedValuesMatrix[i]}
		
		local _, clusterIndexArray = AqwamMatrixLibrary:findMaximumValue(calculatedValuesVector)

		if (clusterIndexArray == nil) then continue end

		local clusterNumber = clusterIndexArray[2]

		clusterVector[i][1] = clusterNumber
		
	end

	return clusterVector

end

function AffinityPropagationModel.new(maxNumberOfIterations, distanceFunction, preferenceType, damping, preferenceValueArray)

	local NewAffinityPropagationModel = BaseModel.new()

	setmetatable(NewAffinityPropagationModel, AffinityPropagationModel)

	NewAffinityPropagationModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewAffinityPropagationModel.distanceFunction = distanceFunction or defaultDistanceFunction
	
	NewAffinityPropagationModel.preferenceType = preferenceType or defaultPreferenceType

	NewAffinityPropagationModel.damping = damping or defaultDamping
	
	NewAffinityPropagationModel.preferenceValueArray = preferenceValueArray

	return NewAffinityPropagationModel

end

function AffinityPropagationModel:setParameters(maxNumberOfIterations, distanceFunction, preferenceType, damping, preferenceValueArray)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.distanceFunction = distanceFunction or self.distanceFunction
	
	self.preferenceType = preferenceType or self.preferenceType

	self.damping = damping or self.damping
	
	self.preferenceValueArray = preferenceValueArray or self.preferenceValueArray
	
end

function AffinityPropagationModel:train(featureMatrix)

	local numberOfData = #featureMatrix
	
	local numberOfIterations = 0
	
	local isConverged = false
	
	local preferenceVector

	local responsibilityMatrix

	local availabilityMatrix

	local clusterVector

	local costArray = {}

	local cost
	
	local ModelParameters = self.ModelParameters
	
	if (ModelParameters) then
		
		responsibilityMatrix = ModelParameters[3]

		availabilityMatrix = ModelParameters[4]
		
	end
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, self.distanceFunction)
	
	local similarityMatrix = AqwamMatrixLibrary:multiply(-1, distanceMatrix)
	
	responsibilityMatrix = responsibilityMatrix or AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)
	
	availabilityMatrix = availabilityMatrix or AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)

	similarityMatrix = setPreferencesToSimilarityMatrix(similarityMatrix, numberOfData, self.preferenceType, self.preferenceValueArray)

	repeat
		
		numberOfIterations += 1
		
		self:iterationWait()

		responsibilityMatrix = calculateResponsibilityMatrix(responsibilityMatrix, availabilityMatrix, similarityMatrix)

		availabilityMatrix = calculateAvailibilityMatrix(responsibilityMatrix, availabilityMatrix, self.damping)
		
		clusterVector = assignClusters(responsibilityMatrix, availabilityMatrix)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(clusterVector, responsibilityMatrix)
			
		end) 
		
		if cost then
			
			table.insert(costArray, cost)

			self:printCostAndNumberOfIterations(cost, numberOfIterations)
			
		end
		
	until (numberOfIterations >= self.maxNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	self.ModelParameters = {featureMatrix, clusterVector, responsibilityMatrix, availabilityMatrix}

	return costArray

end

function AffinityPropagationModel:predict(featureMatrix)

	local maxSimilarityVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	local predictedClusterVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	local storedFeatureMatrix, clusterVector = table.unpack(self.ModelParameters)
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, storedFeatureMatrix, self.distanceFunction)
	
	for i = 1, #featureMatrix, 1 do
		
		local distanceVector = {distanceMatrix[i]}
		
		local _, index = AqwamMatrixLibrary:findMinimumValue(distanceVector)
		
		if (index == nil) then continue end
		
		local storedFeatureMatrixRowIndex = index[2]
		
		predictedClusterVector[i][1] = clusterVector[storedFeatureMatrixRowIndex][1]
		
		maxSimilarityVector[i][1] = distanceVector[1][storedFeatureMatrixRowIndex]
		
	end
	
	return predictedClusterVector, maxSimilarityVector

end

return AffinityPropagationModel
