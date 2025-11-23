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

local OrderingPointsToIdentifyClusteringStructureModel = {}

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

local function calculateReachabilityCost(reachabilityDistanceArray)
	
	local total = 0
	
	local count = 0

	for _, distance in pairs(reachabilityDistanceArray) do
		
		if (distance) and (distance ~= math.huge) then
			
			total = total + distance
			
			count = count + 1
			
		end
		
	end

	return ((count > 0) and (total / count)) or math.huge
end

local function insertSorted(seedPointArray, seedReachabilityDistanceArray, point, distance)
	
	local inserted = false
	
	for i, reachabilityDistance in ipairs(seedReachabilityDistanceArray) do
		
		if (distance < reachabilityDistance) then
			
			table.insert(seedReachabilityDistanceArray, i, distance)
			
			table.insert(seedPointArray, i, point)
			
			inserted = true
			
			break
			
		end
		
	end
	
	if (not inserted) then
		
		table.insert(seedReachabilityDistanceArray, distance)
		
		table.insert(seedPointArray, point)
		
	end
	
end

local function update(neighbourArray, pPointNumber, hasProcessedArray, reachabilityDistanceArray, seedPointArray, seedReachabilityDistanceArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
	local coreDistance = calculateCoreDistance(pPointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunction)
	
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
					
					-- Moving up the new reachability distance inside the seedArray.
					
					insertSorted(seedPointArray, seedReachabilityDistanceArray, oPointNumber, newReachabilityDistance)
					
				end
				
			end
			
		end
		
	end
	
end

local function createClusterArrayArray(orderedPointArray, reachabilityDistanceArray, epsilonPrime)
	
	local clusterArrayArray = {}
	
	local clusterArray = {}
	
	local reachabilityDistance

	for i, pointIndex in ipairs(orderedPointArray) do
		
		reachabilityDistance = reachabilityDistanceArray[pointIndex] or math.huge

		if (reachabilityDistance <= epsilonPrime) then
			
			table.insert(clusterArray, pointIndex)
			
		else
			
			if (#clusterArray > 0) then
				
				table.insert(clusterArrayArray, clusterArray)
				
				clusterArray = {}
				
			end
			
		end
		
	end

	if (#clusterArray > 0) then
		
		table.insert(clusterArrayArray, clusterArray)
		
	end

	return clusterArrayArray
	
end

function OrderingPointsToIdentifyClusteringStructureModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewOrderingPointsToIdentifyClusteringStructureModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewOrderingPointsToIdentifyClusteringStructureModel, OrderingPointsToIdentifyClusteringStructureModel)
	
	NewOrderingPointsToIdentifyClusteringStructureModel:setName("OrderingPointsToIdentifyClusteringStructure")
	
	local epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	local epsilonPrime = (parameterDictionary.epsilonPrime) or (epsilon * 0.5)
	
	NewOrderingPointsToIdentifyClusteringStructureModel.minimumNumberOfPoints = parameterDictionary.minimumNumberOfPoints or defaultMinimumNumberOfPoints
	
	NewOrderingPointsToIdentifyClusteringStructureModel.epsilon = epsilon
	
	NewOrderingPointsToIdentifyClusteringStructureModel.epsilonPrime = epsilonPrime

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
	
	local epsilonPrime = self.epsilonPrime
	
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
			
			if (not reachabilityDistanceArray[pPointNumber]) then
				
				reachabilityDistanceArray[pPointNumber] = math.huge
				
			end
			
			coreDistance = calculateCoreDistance(pPointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)

			if (coreDistance) then
				
				seedPointArray = {}
				
				seedReachabilityDistanceArray = {}
				
				update(neighbourArray, pPointNumber, hasProcessedArray, reachabilityDistanceArray, seedPointArray, seedReachabilityDistanceArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)
				
				for _, qPointNumber in ipairs(seedPointArray) do
					
					neighbourComplementArray = getNeighborArray(qPointNumber, featureMatrix, epsilon, distanceFunctionToApply)
					
					hasProcessedArray[qPointNumber] = true
					
					table.insert(orderedPointArray, qPointNumber)
					
					coreDistance = calculateCoreDistance(qPointNumber, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)
					
					if (coreDistance) then
						
						update(neighbourComplementArray, qPointNumber, hasProcessedArray, reachabilityDistanceArray, seedPointArray, seedReachabilityDistanceArray, featureMatrix, epsilon, minimumNumberOfPoints, distanceFunctionToApply)
						
					end
					
				end
				
			end

		end

		cost = self:calculateCostWhenRequired(pPointNumber, function()

			return calculateReachabilityCost(reachabilityDistanceArray)

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
	
	local clusterArrayArray = createClusterArrayArray(orderedPointArray, reachabilityDistanceArray, epsilonPrime)
	
	self.ModelParameters = {featureMatrix, orderedPointArray, reachabilityDistanceArray, clusterArrayArray}
	
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
	
	local storedFeatureVector, _, _, clusterArrayArray = table.unpack(ModelParameters)
	
	local closestCluster
	
	local shortestDistance
	
	local featureVector
	
	for i, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		closestCluster = nil

		shortestDistance = math.huge

		featureVector = {unwrappedFeatureVector}

		for clusterNumber, clusterArray in ipairs(clusterArrayArray) do

			local distance = 0
			
			for j, pointNumber in ipairs(clusterArray) do

				distance = distance + distanceFunctionToApply(featureVector, {storedFeatureVector[pointNumber]})
				
			end

			distance = distance / #clusterArray

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
