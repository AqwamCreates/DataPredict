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

local BaseModel = require(script.Parent.BaseModel)

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

LocalOutlierFactor = {}

LocalOutlierFactor.__index = LocalOutlierFactor

setmetatable(LocalOutlierFactor, BaseModel)

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local defaultMaximumNumberOfData = math.huge

local function createDistanceMatrix(distanceFunction, featureMatrix)

	local numberOfData = #featureMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfData}, 0)

	local calculateDistance = distanceFunctionDictionary[distanceFunction]
	
	for i, primaryUnwrappedFeatureVector in ipairs(featureMatrix) do
		
		for j, secondaryUnwrappedFeatureVector in ipairs(featureMatrix) do

			distanceMatrix[i][j] = calculateDistance({primaryUnwrappedFeatureVector}, {secondaryUnwrappedFeatureVector})

		end
		
	end

	return distanceMatrix

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function merge(unwrappedDistanceVector, left, mid, right)

	local subArrayOne = mid - left + 1
	local subArrayTwo = right - mid

	local unwrappedLeftDistanceVector = {}
	local unwrappedRightDistanceVector = {}

	for i = 1, subArrayOne do

		unwrappedLeftDistanceVector[i] = unwrappedDistanceVector[left + i - 1]

	end

	for j = 1, subArrayTwo do

		unwrappedRightDistanceVector[j] = unwrappedDistanceVector[mid + j]

	end

	local indexOfSubArrayOne = 1
	local indexOfSubArrayTwo = 1
	local indexOfMergedArray = left

	while indexOfSubArrayOne <= subArrayOne and indexOfSubArrayTwo <= subArrayTwo do

		if unwrappedLeftDistanceVector[indexOfSubArrayOne] <= unwrappedRightDistanceVector[indexOfSubArrayTwo] then

			unwrappedDistanceVector[indexOfMergedArray] = unwrappedLeftDistanceVector[indexOfSubArrayOne]
			indexOfSubArrayOne = indexOfSubArrayOne + 1

		else

			unwrappedDistanceVector[indexOfMergedArray] = unwrappedRightDistanceVector[indexOfSubArrayTwo]
			indexOfSubArrayTwo = indexOfSubArrayTwo + 1

		end

		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayOne <= subArrayOne) do

		unwrappedDistanceVector[indexOfMergedArray] = unwrappedLeftDistanceVector[indexOfSubArrayOne]
		indexOfSubArrayOne = indexOfSubArrayOne + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayTwo <= subArrayTwo) do

		unwrappedDistanceVector[indexOfMergedArray] = unwrappedRightDistanceVector[indexOfSubArrayTwo]
		indexOfSubArrayTwo = indexOfSubArrayTwo + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

end

local function mergeSort(distanceVector, startingValue, endValue)

	if (startingValue >= endValue) then return end

	local mid = math.floor(startingValue + (endValue - startingValue) / 2)

	mergeSort(distanceVector, startingValue, mid)
	mergeSort(distanceVector, mid + 1, endValue)
	merge(distanceVector, startingValue, mid, endValue)

end

function LocalOutlierFactor.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewLocalOutlierFactor = BaseModel.new(parameterDictionary)

	setmetatable(NewLocalOutlierFactor, LocalOutlierFactor)
	
	NewLocalOutlierFactor:setName("LocalOutlierFactor")

	NewLocalOutlierFactor.kValue = parameterDictionary.kValue or defaultKValue

	NewLocalOutlierFactor.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewLocalOutlierFactor.maximumNumberOfData = parameterDictionary.maximumNumberOfData or defaultMaximumNumberOfData
	
	return NewLocalOutlierFactor

end

function LocalOutlierFactor:train(featureMatrix)

	local numberOfData = #featureMatrix

	local maximumNumberOfData = self.maximumNumberOfData

	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		local storedFeatureMatrix = ModelParameters[1]

		local storedLabelVector = ModelParameters[2]

		featureMatrix = AqwamTensorLibrary:concatenate(featureMatrix, storedFeatureMatrix, 1)

		numberOfData = #featureMatrix

		if (numberOfData > maximumNumberOfData) then

			local newFeatureMatrix = {}

			local newLabelVector = {}

			local dataShiftIndex = (numberOfData - maximumNumberOfData)

			for dataIndex = 1, maximumNumberOfData, 1 do

				newFeatureMatrix[dataIndex] = featureMatrix[dataIndex + dataShiftIndex]

			end

			featureMatrix = newFeatureMatrix

			numberOfData = maximumNumberOfData

		end

	end

	if (numberOfData < self.kValue) and (self.isOutputPrinted) then warn("Number of data is less than the K value. Please add more data before doing any predictions.") end

	self.ModelParameters = featureMatrix

end

function LocalOutlierFactor:score()
	
	local featureMatrix = self.ModelParameters

	if (not featureMatrix) then return {{math.huge}} end

	local kValue = self.kValue

	local distanceFunction = self.distanceFunction
	
	local numberOfData = #featureMatrix
	
	local numberOfDataMinusOne = numberOfData - 1
	
	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix)
	
	local nearestNeighbourIndexArrayArray = {}
	
	local numberOfNearestNeighboursVector = {}
	
	local reachabilityDistanceMatrix = {}
	
	for i, unwrappedDistanceVector in ipairs(distanceMatrix) do

		local sortedUnwrappedDistanceVector = deepCopyTable(unwrappedDistanceVector)
		
		table.remove(sortedUnwrappedDistanceVector, i) -- Ignore the distance to itself.

		mergeSort(sortedUnwrappedDistanceVector, 1, numberOfDataMinusOne)
		
		local kDistance = sortedUnwrappedDistanceVector[kValue]
		
		local nearestNeighbourIndexArray = {}
		
		local unwrappedReachabilityDistanceVector = {}
		
		local numberOfNearestNeighbours = 0
		
		for j, distance in ipairs(unwrappedDistanceVector) do
			
			unwrappedReachabilityDistanceVector[j] = math.max(kDistance, distance)
			
			if (distance <= kDistance) then table.insert(nearestNeighbourIndexArray, j) end
			
		end
		
		reachabilityDistanceMatrix[i] = unwrappedReachabilityDistanceVector
		
		nearestNeighbourIndexArrayArray[i] = nearestNeighbourIndexArray
		
		numberOfNearestNeighboursVector[i] = {#nearestNeighbourIndexArray}

	end
	
	local sumReachabilityDistanceVector = AqwamTensorLibrary:sum(reachabilityDistanceMatrix, 2)
	
	local localReachabilityDensityVector = AqwamTensorLibrary:divide(numberOfNearestNeighboursVector, sumReachabilityDistanceVector)
	
	local localOutlierFactorVectorDivisor = AqwamTensorLibrary:multiply(numberOfNearestNeighboursVector, localReachabilityDensityVector)
	
	local localOutlierFactorVectorNumerator = {}

	for i, nearestNeighboursIndexArray in ipairs(nearestNeighbourIndexArrayArray) do
		
		local sumRatio = 0
		
		for _, nearestNeighbourIndex in ipairs(nearestNeighboursIndexArray) do
			
			sumRatio = sumRatio + localReachabilityDensityVector[nearestNeighbourIndex][1]
			
		end
		
		localOutlierFactorVectorNumerator[i] = {sumRatio}
		
	end
	
	local localOutlierFactorVector = AqwamTensorLibrary:divide(localOutlierFactorVectorNumerator, localOutlierFactorVectorDivisor)
	
	return localOutlierFactorVector

end

return LocalOutlierFactor
