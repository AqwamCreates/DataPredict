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

OrderingPointsToIdentifyClusteringStructureModel = {}

OrderingPointsToIdentifyClusteringStructureModel.__index = OrderingPointsToIdentifyClusteringStructureModel

setmetatable(OrderingPointsToIdentifyClusteringStructureModel, IterativeMethodBaseModel)

local defaultMinimumNumberOfPoints = 2

local defaultDistanceFunction = "Manhattan"

local defaultEpsilon = 10

local function getNeighborArray(currentCorePointNumber, featureMatrix, epsilon, distanceFunction)
	
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

local function calculateCoreDistance(pointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	local distanceArray = {}
	
	local distance
	
	for i, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		if (i ~= pointNumber) then

			distance = distanceFunction({featureMatrix[pointNumber]}, {unwrappedFeatureVector})

			table.insert(distanceArray, distance)

		end
		
	end
	
	if (#distanceArray < minimumNumberOfPoints) then return end
	
	table.sort(distanceArray)
	
	return distanceArray[minimumNumberOfPoints]
	
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

local function update(neighbourArray, pPointNumber, seedPointArray, seedReachabilityDistanceArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	local coreDistance = calculateCoreDistance(pPointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	local hasProcessedArray = {}
	
	local reachabilityDistanceArray = {}
	
	local reachabilityDistance
	
	local newReachabilityDistance
	
	local newReachabilityDistanceArrayIndex
	
	for _, oPointNumber in ipairs(neighbourArray) do
		
		if (not hasProcessedArray[oPointNumber]) then
			
			newReachabilityDistance = math.max(coreDistance, distanceFunction({featureMatrix[pPointNumber]}, {featureMatrix[oPointNumber]}))
			
			reachabilityDistance = reachabilityDistanceArray[oPointNumber]
			
			if (not reachabilityDistance) then
				
				reachabilityDistanceArray[oPointNumber] = newReachabilityDistance
				
				table.insert(seedPointArray, oPointNumber)
				
				table.insert(seedReachabilityDistanceArray, newReachabilityDistance)
				
			else
				
				if (newReachabilityDistance < reachabilityDistance) then
					
					reachabilityDistanceArray[oPointNumber] = newReachabilityDistance
					
					newReachabilityDistanceArrayIndex = table.find(seedReachabilityDistanceArray, newReachabilityDistance)
					
					-- Moving up the new reachability distance inside the seedArray.
					
					table.remove(seedPointArray, newReachabilityDistanceArrayIndex)
					
					table.remove(seedReachabilityDistanceArray, newReachabilityDistanceArrayIndex)
					
					newReachabilityDistanceArrayIndex = newReachabilityDistanceArrayIndex - 1
					
					table.insert(seedPointArray, newReachabilityDistanceArrayIndex, oPointNumber)
					
					table.insert(seedReachabilityDistanceArray, newReachabilityDistanceArrayIndex, newReachabilityDistance)
					
				end
				
			end
			
		end
		
	end
	
end

function OrderingPointsToIdentifyClusteringStructureModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewOrderingPointsToIdentifyClusteringStructureModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewOrderingPointsToIdentifyClusteringStructureModel, OrderingPointsToIdentifyClusteringStructureModel)
	
	NewOrderingPointsToIdentifyClusteringStructureModel:setName("OrderingPointsToIdentifyClusteringStructure")
	
	NewOrderingPointsToIdentifyClusteringStructureModel.minimumNumberOfPoints = parameterDictionary.minimumNumberOfPoints or defaultMinimumNumberOfPoints
	
	NewOrderingPointsToIdentifyClusteringStructureModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewOrderingPointsToIdentifyClusteringStructureModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	return NewOrderingPointsToIdentifyClusteringStructureModel
	
end

function OrderingPointsToIdentifyClusteringStructureModel:train(featureMatrix)
	
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
	
	local reachabilityDistanceArray = {}
	
	local hasProcessedArray = {}
	
	local orderedPointArray = {}
	
	local neighbourArray
	
	local coreDistance
	
	local seedPointArray
	
	local seedReachabilityDistanceArray
	
	local neighbourComplementArray
	
	local cost
	
	for pPointNumber, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		self:iterationWait()

		if (not hasProcessedArray[pPointNumber]) then
			
			neighbourArray = getNeighborArray(pPointNumber, featureMatrix, epsilon, distanceFunctionToApply)

			hasProcessedArray[pPointNumber] = true
			
			table.insert(orderedPointArray, pPointNumber)
			
			coreDistance = calculateCoreDistance(pPointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)

			if (coreDistance) then
				
				seedPointArray = {}
				
				seedReachabilityDistanceArray = {}
				
				update(neighbourArray, pPointNumber, seedPointArray, seedReachabilityDistanceArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)
				
				for _, qPointNumber in ipairs(seedPointArray) do
					
					neighbourComplementArray = getNeighborArray(qPointNumber, featureMatrix, epsilon, distanceFunctionToApply)
					
					hasProcessedArray[qPointNumber] = true
					
					table.insert(orderedPointArray, qPointNumber)
					
					coreDistance = calculateCoreDistance(qPointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)
					
					if (coreDistance) then
						
						update(neighbourComplementArray, qPointNumber, seedPointArray, seedReachabilityDistanceArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)
						
					end
					
				end
				
			end

		end

		cost = self:calculateCostWhenRequired(pPointNumber, function()

			return calculateCost(distanceFunctionToApply, featureMatrix)

		end)

		if (cost) then

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(pPointNumber, cost)

			if self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) then break end

		end
		
	end
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	self.ModelParameters = {featureMatrix}
	
	return costArray
	
end

function OrderingPointsToIdentifyClusteringStructureModel:predict(featureMatrix)
	
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

return OrderingPointsToIdentifyClusteringStructureModel
