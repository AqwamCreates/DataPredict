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

FuzzyCMeansModel = {}

FuzzyCMeansModel.__index = FuzzyCMeansModel

setmetatable(FuzzyCMeansModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultNumberOfClusters = 1

local defaultFuzziness = 2

local defaultDistanceFunction = "Euclidean"

local defaultMode = "Hybrid"

local defaultSetInitialCentroidsOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = true

local defaultEpsilon = 1e-16

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

		local x1Magnitude = math.sqrt(x1MagnitudePart2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

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

local function checkIfTheDataPointClusterNumberBelongsToTheCluster(dataPointClusterNumber, cluster)
	
	if (dataPointClusterNumber == cluster) then
		
		return 1
		
	else
		
		return 0
		
	end
	
end

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

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
	
	local centroidMatrix = {}
	
	local dataIndexArray = {}
	
	local dataIndex
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, distanceFunction)
	
	repeat
		
		dataIndex = chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, dataIndexArray)
		
		table.insert(dataIndexArray, dataIndex)
		
	until (#dataIndexArray == numberOfClusters)
	
	for row = 1, numberOfClusters, 1 do
		
		dataIndex = dataIndexArray[row]
		
		table.insert(centroidMatrix, featureMatrix[dataIndex])
		
	end
	
	return centroidMatrix
	
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

local function calculateCost(distanceMatrix, clusterMembershipMatrix)
	
	local costMatrix = AqwamTensorLibrary:multiply(distanceMatrix, clusterMembershipMatrix)
	
	local cost = AqwamTensorLibrary:sum(costMatrix)
	
	return cost
	
end

function FuzzyCMeansModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewFuzzyCMeansModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewFuzzyCMeansModel, FuzzyCMeansModel)
	
	NewFuzzyCMeansModel:setName("FuzzyCMeans")

	NewFuzzyCMeansModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters
	
	NewFuzzyCMeansModel.fuzziness = NewFuzzyCMeansModel:getValueOrDefaultValue(parameterDictionary.fuzziness, defaultFuzziness)
	
	NewFuzzyCMeansModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewFuzzyCMeansModel.mode = parameterDictionary.mode or defaultMode

	NewFuzzyCMeansModel.setInitialCentroidsOnDataPoints =  NewFuzzyCMeansModel:getValueOrDefaultValue(parameterDictionary.setInitialCentroidsOnDataPoints, defaultSetInitialCentroidsOnDataPoints)
	
	NewFuzzyCMeansModel.setTheCentroidsDistanceFarthest = NewFuzzyCMeansModel:getValueOrDefaultValue(parameterDictionary.setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	NewFuzzyCMeansModel.epsilon = NewFuzzyCMeansModel:getValueOrDefaultValue(parameterDictionary.epsilon, defaultEpsilon)
	
	return NewFuzzyCMeansModel
	
end

function FuzzyCMeansModel:initializeCentroids(featureMatrix, numberOfClusters, distanceFunction)
	
	local setInitialCentroidsOnDataPoints = self.setInitialCentroidsOnDataPoints
	
	local setTheCentroidsDistanceFarthest = self.setTheCentroidsDistanceFarthest
	
	if (setInitialCentroidsOnDataPoints) and (numberOfClusters == 1) then
		
		return AqwamTensorLibrary:mean(featureMatrix, 1)
	
	elseif (setInitialCentroidsOnDataPoints) and (setTheCentroidsDistanceFarthest) then

		return chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	elseif (setInitialCentroidsOnDataPoints) and (not setTheCentroidsDistanceFarthest) then

		return chooseRandomCentroids(featureMatrix, numberOfClusters)

	else

		return self:initializeMatrixBasedOnMode({numberOfClusters, #featureMatrix[1]})

	end
	
end

local function calculateMembershipMatrix(distanceMatrix, fuzziness, epsilon)
	
	local numberOfData = #distanceMatrix
	
	local numberOfClusters = #distanceMatrix[1]
	
	local membershipMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfClusters}, 0)
	
	local ratioPowerConstant = 2 / (fuzziness - 1)
	
	for dataIndex, unwrappedDistanceVector in ipairs(distanceMatrix) do
		
		for i = 1, numberOfClusters do

			local denominator = 0
			
			local distanceI = unwrappedDistanceVector[i]

			for j = 1, numberOfClusters do
				
				local distanceJ = unwrappedDistanceVector[j]
				
				if (distanceJ ~= 0) then
					
					local ratio = distanceI / distanceJ
					
					denominator = denominator + (ratio ^ ratioPowerConstant)
					
				end

			end

			membershipMatrix[dataIndex][i] = 1 / (denominator + epsilon)

		end
		
	end

	return membershipMatrix
end

local function calculateMean(featureMatrix, centroidMatrix, clusterMembershipMatrix, fuzziness, epsilon)

	local numberOfData = #featureMatrix

	local numberOfCentroids = #centroidMatrix

	local numberOfFeatures = #centroidMatrix[1]

	local sumOfAssignedCentroidVector = AqwamTensorLibrary:sum(clusterMembershipMatrix, 1) -- since row is the number of data in clusterAssignmentMatrix, then we vertical sum it

	local newCentroidMatrix = AqwamTensorLibrary:createTensor({numberOfCentroids, numberOfFeatures})

	for cluster = 1, numberOfCentroids, 1 do

		local numeratorVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 0)

		local denominator = 0

		for dataIndex, unwrappedDataVector in ipairs(featureMatrix) do

			local membershipValue = clusterMembershipMatrix[dataIndex][cluster]^fuzziness

			local multipliedMembershipValue = AqwamTensorLibrary:multiply({unwrappedDataVector}, membershipValue)

			numeratorVector = AqwamTensorLibrary:add(numeratorVector, multipliedMembershipValue)

			denominator = denominator + membershipValue

		end

		local newCentroidVector = AqwamTensorLibrary:divide(numeratorVector, (denominator + epsilon))

		newCentroidMatrix[cluster] = newCentroidVector[1]

	end

	return newCentroidMatrix

end

local function calculateMatrices(featureMatrix, centroidMatrix, distanceMatrix, fuzziness, epsilon)

	local clusterMembershipMatrix = calculateMembershipMatrix(distanceMatrix, fuzziness, epsilon)
	
	centroidMatrix = calculateMean(featureMatrix, centroidMatrix, clusterMembershipMatrix, fuzziness, epsilon)
	
	return centroidMatrix, clusterMembershipMatrix
	
end

function FuzzyCMeansModel:train(featureMatrix)
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local fuzziness = self.fuzziness
	
	local distanceFunction = self.distanceFunction
	
	local mode = self.mode
	
	local epsilon = self.epsilon
	
	local centroidMatrix = self.ModelParameters
	
	local clusterMembershipMatrix
	
	local distanceMatrix

	if (mode == "Hybrid") then -- This must be always above the centroid initialization check. Otherwise it will think this is second training round despite it being the first one!
		
		mode = (centroidMatrix and "Online") or "Offline"

	end
	
	local distanceFunctionToApply = distanceFunctionList[distanceFunction]

	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	if (mode == "Offline") then centroidMatrix = nil end
	
	if (centroidMatrix) then
		
		if (#featureMatrix[1] ~= #centroidMatrix[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		centroidMatrix = self:initializeCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply)
		
	end

	local numberOfIterations = 0
	
	local costArray = {}
	
	local cost
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, distanceFunctionToApply)

		centroidMatrix, clusterMembershipMatrix = calculateMatrices(featureMatrix, centroidMatrix, distanceMatrix, fuzziness, epsilon)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return calculateCost(distanceMatrix, clusterMembershipMatrix)

		end)
		
		if (cost) then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.ModelParameters = centroidMatrix
	
	return costArray
	
end

function FuzzyCMeansModel:predict(featureMatrix, returnMode)
	
	local distanceFunctionToApply = distanceFunctionList[self.distanceFunction]
	
	local centroidMatrix = self.ModelParameters
	
	local distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, distanceFunctionToApply)
	
	local returnType = type(returnMode)
	
	if (returnType ~= "nil") then
		
		local isBoolean = (returnType == "boolean")
		
		if (returnMode == "Distance") or (isBoolean and returnMode) then
			
			return distanceMatrix
		
		elseif (returnMode == "Membership") then
			
			return calculateMembershipMatrix(distanceMatrix, self.fuzziness, self.epsilon)
			
		else
			
			error("Unknown return mode value.")
			
		end
		
	end
	
	local clusterNumberVector, clusterDistanceVector = assignToCluster(distanceMatrix)

	return clusterNumberVector, clusterDistanceVector
	
end

return FuzzyCMeansModel
